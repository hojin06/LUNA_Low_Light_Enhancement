"""LUNA detection-aware fine-tuning — frozen YOLOv8n teacher 기반 추가 학습.

배경 (Why)
----------
LUNA-LoLI-30K 체크포인트는 ExDark full 7,363 장 mAP@0.5 = 0.346 으로
원본 ExDark (0.447) 대비 부족.  PSNR/SSIM 만 보던 기존 reconstruction loss
로는 *YOLO 가 잘 보는 이미지* 가 만들어진다는 보장이 없으므로, frozen
YOLOv8n backbone+neck feature 를 GT 와 정합시키는 **detection-aware loss**
를 추가해 detection 성능을 직접 끌어올린다.

학습 구성 (How)
---------------
* Stage 1 만 (adversarial 없음, 30 epochs, lr=1e-5, AdamW, gradient clip=1.0).
* 데이터: LoLI-Street 4k + LOL-v2 Real 1k = 5k mixed (train_loli_street 의
  ``build_mixed_train_loader`` 재사용).
* 평가: LoLI val 500 / LOL eval15 / ExDark quick eval (every 5 epoch).
* Loss:
      L_total = L_rec + λ_det_eff · L_detection_aware
      L_rec   = L1 + 0.5·L_vgg + 1.0·L_ssim                  (SupervisedLoss)
      L_det   = L_feat_mse + α·L_feat_cos + β·L_preserve     (DetectionAwareLoss)
* **λ_det warmup**: 처음 ``--warmup_epochs`` (기본 5) 는 0 으로 reconstruction
  만 학습 → 이후 선형 증가하여 ``--lambda_det`` 도달.  YOLO feature gradient
  가 초기에 LUNA 를 흔드는 것 방지.
* **YOLO frozen**: ``requires_grad=False`` + 항상 eval.  단, forward 는
  ``torch.no_grad`` 로 감싸지 않아 LUNA 까지 grad 가 흐른다.

체크포인트
----------
* ``checkpoints/det_aware_{tag}_best_psnr.pth`` — LoLI val PSNR 기준 best.
* ``checkpoints/det_aware_{tag}_best_map.pth``  — ExDark quick mAP@0.5 기준 best.
* ``checkpoints/det_aware_{tag}_last.pth``      — epoch-by-epoch resume 용.
  ``tag = {001, 005, 005_preserve, ...}``  (λ_det × 100 zero-pad 3, + preserve 옵션).

실행
----
    python train_detection_aware.py --lambda_det 0.01 --max_samples 5000 \\
        --resume checkpoints/loli_street_30000_stage2_best.pth \\
        --loli_root "...\\LoLI-Street" --lol_root "...\\LOL-v2"
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows 콘솔 UTF-8 (한글 출력)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SupervisedLoss
from utils import evaluate, psnr_metric, save_comparison_grid
from utils.detection_aware_loss import DetectionAwareLoss
from utils.yolo_features import YOLOFeatureExtractor

# 기존 인프라 재사용 (수정하지 않음)
from train import set_seed
from train_hybrid_v1_final import (
    HYBRID_V1_BASE_FILTERS, HYBRID_V1_CONV_CONFIG,
    HYBRID_V1_USE_ATTENTION, build_hybrid_v1_generator,
)
from train_loli_street import (
    _load_g_weights, _measure_baseline_eval15,
    build_eval_loaders, build_mixed_train_loader,
)


HRULE = "=" * 100
SUBRULE = "-" * 100


# ===========================================================================
# 1. λ_det warmup — 첫 W 에폭 0, 이후 선형 증가하여 target 도달
# ===========================================================================
def lambda_det_for_epoch(
    epoch: int, target: float, warmup_epochs: int,
) -> float:
    """epoch 1-indexed.  ``warmup_epochs=5`` 이면 ep 1~5 는 0, ep 6 부터 선형 증가.

    구간 (W = warmup_epochs):
        epoch ≤ W            → 0.0
        W < epoch ≤ 2W       → target · (epoch - W) / W   (선형 증가)
        epoch > 2W           → target                        (full)
    """
    if warmup_epochs <= 0:
        return float(target)
    if epoch <= warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / float(warmup_epochs)
    return float(target) * min(progress, 1.0)


# ===========================================================================
# 2. 체크포인트 helper
# ===========================================================================
def _save_ckpt(
    path: Path, epoch: int,
    G: nn.Module, opt_g, sch_g,
    best_psnr: float, best_map: float,
    args: argparse.Namespace,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    state: Dict[str, Any] = {
        "epoch": epoch,
        "generator": G.state_dict(),
        "opt_g":     opt_g.state_dict(),
        "sch_g":     sch_g.state_dict(),
        "best_psnr": float(best_psnr),
        "best_map":  float(best_map),
        "base_filters":  HYBRID_V1_BASE_FILTERS,
        "use_attention": HYBRID_V1_USE_ATTENTION,
        "conv_config":   HYBRID_V1_CONV_CONFIG.copy(),
        "dataset":       "loli_street_mix",
        "lambda_det":    float(args.lambda_det),
        "alpha_cos":     float(args.alpha_cos),
        "beta_preserve": float(args.beta_preserve),
        "use_preserve":  bool(args.use_preserve),
        "warmup_epochs": int(args.warmup_epochs),
    }
    if extra:
        state.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ===========================================================================
# 3. CSV 로거 — det-aware 전용 컬럼 포함
# ===========================================================================
CSV_FIELDS: List[str] = [
    "run_tag", "epoch", "lr",
    "lambda_det_effective",
    "L_total", "L_rec", "L_l1", "L_vgg", "L_ssim",
    "L_feat_mse", "L_feat_cos", "L_preserve",
    "L_mse_p3", "L_mse_p4", "L_mse_p5",
    "L_cos_p3", "L_cos_p4", "L_cos_p5",
    "train_psnr",
    "loli_val_psnr", "loli_val_ssim",
    "lol_eval15_psnr", "lol_eval15_ssim",
    "delta_eval15_psnr",
    "exdark_map50", "exdark_map", "exdark_p", "exdark_r",
    "best_psnr", "best_map",
    "grad_norm_mean", "epoch_sec",
]


class DetCsvLogger:
    """append-mode CSV.  여러 experiment 가 같은 파일을 공유하므로 ``run_tag`` 컬럼 포함."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.stat().st_size == 0:
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        full = {k: row.get(k, "") for k in CSV_FIELDS}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(full)


# ===========================================================================
# 4. Sample image saving — [low | LUNA enhanced | GT high] 가로 비교
# ===========================================================================
def save_epoch_samples(
    G: nn.Module,
    loader: DataLoader,
    out_dir: Path, epoch: int,
    device: str, num_samples: int = 4,
) -> None:
    """``loader`` 앞쪽 ``num_samples`` 개 페어로 비교 PNG 생성.

    저장 위치: ``out_dir / epoch_XXX_sample_NNN.png``,
    레이아웃: 가로 [low | LUNA enhanced | GT high].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    G.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            low, high = batch
            low  = low.to(device,  non_blocking=True)
            high = high.to(device, non_blocking=True)
            fake = G(low)
            for i in range(low.size(0)):
                if saved >= num_samples:
                    break
                out_path = out_dir / f"epoch_{epoch:03d}_sample_{saved+1:03d}.png"
                save_comparison_grid(
                    [low[i].detach(), fake[i].detach(), high[i].detach()],
                    out_path, ncols=3, pad=4,
                )
                saved += 1
            if saved >= num_samples:
                break


# ===========================================================================
# 5. ExDark quick eval — N 장 sample 로 mAP@0.5 / mAP / P / R 측정
# ===========================================================================
def quick_exdark_eval(
    G: nn.Module,
    exdark_root: Path,
    yolo_predict_model: Any,    # ultralytics YOLO instance (별도 — feature 추출기와 분리)
    image_size: int,
    device: str,
    num_samples: int = 100,
    conf: float = 0.25,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    """ExDark Test split 에서 ``num_samples`` 장 sampling 후 quick mAP 측정.

    Returns
    -------
    dict | None
        ``{"map50", "map", "p", "r", "n_images"}``.
        ExDark 미존재 / sample 0 인 경우 ``None`` 반환.
    """
    try:
        from experiments.downstream_exdark import (  # type: ignore
            DetectionAccumulator, collect_exdark_samples, parse_bbgt_v3,
            enhance_with_luna, _run_yolo, _extract_boxes,
        )
    except ImportError as e:
        print(f"  [exdark-eval] import 실패: {e}")
        return None

    if not (exdark_root / "images").is_dir():
        return None

    samples = collect_exdark_samples(exdark_root, splits=(3,))
    if not samples:
        return None
    rng = random.Random(seed)
    if num_samples > 0 and num_samples < len(samples):
        samples = rng.sample(samples, k=num_samples)

    acc = DetectionAccumulator()
    G.eval()
    with torch.no_grad():
        for img_id, sm in enumerate(samples):
            try:
                pil = Image.open(sm.image_path).convert("RGB")
            except Exception:
                continue
            recs = parse_bbgt_v3(sm.ann_path)
            if recs:
                gt_boxes = np.array([[x1, y1, x2, y2] for (_c, x1, y1, x2, y2) in recs],
                                    dtype=np.float32)
                gt_classes = np.array([c for (c, *_r) in recs], dtype=np.int64)
            else:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
                gt_classes = np.zeros((0,), dtype=np.int64)
            enh_rgb = enhance_with_luna(G, pil, image_size, device)
            r_enh = _run_yolo(yolo_predict_model, enh_rgb, conf, device)
            pb, pc, pf = _extract_boxes(r_enh)
            acc.add(img_id, gt_boxes, gt_classes, pb, pc, pf)

    res = acc.compute()
    return {
        "map50":    float(res["map50"]),
        "map":      float(res["map"]),
        "p":        float(res["p50"]),
        "r":        float(res["r50"]),
        "n_images": int(res["n_images"]),
    }


# ===========================================================================
# 6. Training loop — Stage 1 only, with detection-aware loss
# ===========================================================================
def train_detection_aware(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    train_loader: DataLoader,
    loli_val_loader: DataLoader,
    lol15_loader: DataLoader,
    baseline_eval15_psnr: Optional[float],
    csv_logger: DetCsvLogger,
    yolo_predict_model: Any,
) -> None:
    device = args.device
    set_seed(args.seed)

    print()
    print(HRULE)
    print(f" ★ Detection-aware fine-tuning  (run_tag = {args.run_tag})")
    print(HRULE)
    print(f"  epochs / lr      : {args.epochs} / {args.lr:.2e}  (AdamW)")
    print(f"  batch_size       : {args.batch_size}")
    print(f"  λ_det target     : {args.lambda_det}  (warmup_epochs = {args.warmup_epochs})")
    print(f"  α_cos / β_pres   : {args.alpha_cos} / {args.beta_preserve}  "
          f"use_preserve = {args.use_preserve}")
    print(f"  grad clip max    : {args.grad_clip_max}")
    print(f"  resume           : {args.resume}")
    print(f"  baseline eval15  : {baseline_eval15_psnr}")
    print(f"  device           : {device}")
    print(SUBRULE)

    # ---- Generator ----
    G = build_hybrid_v1_generator().to(device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"  G parameters     : {n_params:,}")

    # ---- YOLO feature extractor (frozen) ----
    print(f"  loading YOLO     : {args.yolo_weights}")
    yolo_feat = YOLOFeatureExtractor(
        weights=args.yolo_weights,
        input_size=args.image_size,
        debug=args.debug_yolo_layers,
    ).to(device)
    yolo_params = sum(p.numel() for p in yolo_feat.parameters())
    yolo_trainable = sum(p.numel() for p in yolo_feat.parameters() if p.requires_grad)
    assert yolo_trainable == 0, "YOLO 가 frozen 이 아닙니다."
    print(f"  YOLO params      : {yolo_params:,} (trainable={yolo_trainable})")
    print(f"  P3/P4/P5 idx     : "
          f"{yolo_feat.p3_idx}/{yolo_feat.p4_idx}/{yolo_feat.p5_idx}  "
          f"shapes = {yolo_feat.feature_shapes()}")

    # ---- Losses ----
    rec_loss = SupervisedLoss(
        lambda_l1=args.lambda_l1,
        lambda_vgg=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim,
    ).to(device)
    det_loss = DetectionAwareLoss(
        yolo_extractor=yolo_feat,
        alpha_cos=args.alpha_cos,
        beta_preserve=args.beta_preserve,
        use_preserve=args.use_preserve,
    ).to(device)

    # ---- Optimizer / Scheduler ----
    opt_g = AdamW(G.parameters(), lr=args.lr,
                  betas=(args.beta1, args.beta2),
                  weight_decay=args.weight_decay)
    sch_g = CosineAnnealingLR(opt_g, T_max=args.epochs,
                              eta_min=args.lr * args.eta_min_ratio)

    # ---- Resume external G weights (only if no _last yet) ----
    if args.resume and not paths["last"].is_file():
        rp = Path(args.resume)
        if rp.is_file():
            s = _load_g_weights(G, rp, device=device)
            print(f"  resume G from    : {rp.name}  "
                  f"(prev best LoLI PSNR = {s.get('best_loli_val_psnr', float('nan'))})")
        else:
            print(f"  [warn] --resume 파일 없음: {rp} — 무작위 초기화로 시작.")

    # ---- Resume from _last (epoch-level) ----
    start_epoch = 1
    best_psnr: float = -float("inf")
    best_map:  float = -float("inf")
    if paths["last"].is_file():
        s = torch.load(paths["last"], map_location=device, weights_only=False)
        G.load_state_dict(s["generator"])
        opt_g.load_state_dict(s["opt_g"])
        sch_g.load_state_dict(s["sch_g"])
        start_epoch = int(s.get("epoch", 0)) + 1
        best_psnr = float(s.get("best_psnr", -float("inf")))
        best_map  = float(s.get("best_map",  -float("inf")))
        print(f"  resume _last     : epoch {start_epoch}, "
              f"best_psnr={best_psnr:.3f}, best_map={best_map:.4f}")
        if start_epoch > args.epochs:
            print("  [info] 이미 완료된 학습 — 건너뜀.")
            return

    # 초기 평가 (resume 이 아닐 때만, best_psnr/best_map 기준점 확립)
    if start_epoch == 1 and best_psnr == -float("inf"):
        init = evaluate(G, loli_val_loader, device=device)
        best_psnr = float(init["psnr"])
        print(f"  init LoLI PSNR   : {best_psnr:.3f}")
        _save_ckpt(paths["best_psnr"], 0, G, opt_g, sch_g,
                   best_psnr, best_map, args)

    print(SUBRULE)
    print(" 학습 시작")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        lam_eff = lambda_det_for_epoch(epoch, args.lambda_det, args.warmup_epochs)

        G.train()
        yolo_feat.eval()  # 안전 재확인
        sums = {
            "total": 0.0, "rec": 0.0, "l1": 0.0, "vgg": 0.0, "ssim": 0.0,
            "feat_mse": 0.0, "feat_cos": 0.0, "preserve": 0.0,
            "mse_p3": 0.0, "mse_p4": 0.0, "mse_p5": 0.0,
            "cos_p3": 0.0, "cos_p4": 0.0, "cos_p5": 0.0,
            "train_psnr": 0.0, "grad_norm": 0.0,
        }
        n = 0

        pbar = tqdm(
            train_loader,
            desc=f"DET Ep {epoch:3d}/{args.epochs} λ={lam_eff:.4f}",
            ncols=130, leave=False,
        )
        for low, high in pbar:
            low  = low.to(device,  non_blocking=True)
            high = high.to(device, non_blocking=True)
            bs = low.size(0)

            opt_g.zero_grad(set_to_none=True)
            fake = G(low)

            rec_out = rec_loss(fake, high)
            l_rec = rec_out["total"]

            if lam_eff > 0.0:
                det_out = det_loss(
                    luna_pm1=fake, high_pm1=high,
                    low_pm1=low if args.use_preserve else None,
                )
                l_det_total = det_out["total"]
                total = l_rec + lam_eff * l_det_total
            else:
                det_out = None
                total = l_rec

            total.backward()
            # gradient clipping — frozen YOLO 통과 grad 의 폭주 방지
            grad_norm = torch.nn.utils.clip_grad_norm_(
                G.parameters(), max_norm=args.grad_clip_max,
            )
            opt_g.step()

            with torch.no_grad():
                tp = psnr_metric(fake.detach(), high)
            sums["total"]      += float(total.detach()) * bs
            sums["rec"]        += float(l_rec.detach()) * bs
            sums["l1"]         += float(rec_out["l1"])  * bs
            sums["vgg"]        += float(rec_out["vgg"]) * bs
            sums["ssim"]       += float(rec_out["ssim"])* bs
            sums["train_psnr"] += float(tp) * bs
            sums["grad_norm"]  += float(grad_norm) * bs
            if det_out is not None:
                sums["feat_mse"] += float(det_out["feat_mse"]) * bs
                sums["feat_cos"] += float(det_out["feat_cos"]) * bs
                sums["preserve"] += float(det_out["preserve"]) * bs
                for level in ("p3", "p4", "p5"):
                    sums[f"mse_{level}"] += float(det_out["per_level_mse"][level]) * bs
                    sums[f"cos_{level}"] += float(det_out["per_level_cos"][level]) * bs
            n += bs

            pbar.set_postfix({
                "L":     f"{float(total):.3f}",
                "rec":   f"{float(l_rec):.3f}",
                "fMSE":  f"{(float(det_out['feat_mse']) if det_out else 0):.4f}",
                "fCOS":  f"{(float(det_out['feat_cos']) if det_out else 0):.4f}",
                "PSNR":  f"{float(tp):.1f}",
                "|g|":   f"{float(grad_norm):.2f}",
            })
        pbar.close()
        sch_g.step()
        epoch_sec = time.perf_counter() - t0

        # ---- 평가 ----
        loli_eval  = evaluate(G, loli_val_loader, device=device)
        lol15_eval = evaluate(G, lol15_loader,   device=device)

        # ExDark quick eval (every N epochs, 마지막 epoch 도 포함)
        exdark_result: Optional[Dict[str, float]] = None
        if args.exdark_root and (
            epoch % max(args.eval_every_exdark, 1) == 0 or epoch == args.epochs
        ):
            print(f"  [ep {epoch:3d}] ExDark quick eval ({args.exdark_num_samples} samples) ...")
            exdark_result = quick_exdark_eval(
                G=G, exdark_root=Path(args.exdark_root),
                yolo_predict_model=yolo_predict_model,
                image_size=args.image_size, device=device,
                num_samples=args.exdark_num_samples,
                conf=args.exdark_conf, seed=args.seed,
            )

        # ---- 콘솔 한 줄 요약 ----
        parts = [
            f"Ep {epoch:3d}/{args.epochs}",
            f"λ_eff={lam_eff:.4f}",
            f"LoLI={loli_eval['psnr']:6.3f}/{loli_eval['ssim']:.4f}",
            f"e15={lol15_eval['psnr']:6.3f}/{lol15_eval['ssim']:.4f}",
        ]
        if baseline_eval15_psnr is not None:
            parts.append(f"Δe15={lol15_eval['psnr']-baseline_eval15_psnr:+.3f}")
        if exdark_result is not None:
            parts.append(f"ExDark mAP@0.5={exdark_result['map50']:.4f} "
                         f"P/R={exdark_result['p']:.3f}/{exdark_result['r']:.3f}")
        parts.append(f"sec={epoch_sec:.1f}")
        print("  " + " | ".join(parts))
        if baseline_eval15_psnr is not None and lol15_eval["psnr"] < args.forget_threshold:
            print(f"    ⚠️  eval15 < {args.forget_threshold} (catastrophic forgetting 위험)")

        # ---- best 갱신 ----
        improved_psnr = loli_eval["psnr"] > best_psnr
        if improved_psnr:
            best_psnr = loli_eval["psnr"]
            _save_ckpt(paths["best_psnr"], epoch, G, opt_g, sch_g,
                       best_psnr, best_map, args)
            print(f"    → new best_psnr = {best_psnr:.3f} (saved {paths['best_psnr'].name})")
        if exdark_result is not None and exdark_result["map50"] > best_map:
            best_map = float(exdark_result["map50"])
            _save_ckpt(paths["best_map"], epoch, G, opt_g, sch_g,
                       best_psnr, best_map, args,
                       extra={"exdark_eval": exdark_result})
            print(f"    → new best_map  = {best_map:.4f} (saved {paths['best_map'].name})")

        # ---- 샘플 이미지 (every N epochs, 마지막 epoch 도 포함) ----
        if epoch % max(args.save_samples_every, 1) == 0 or epoch == args.epochs:
            save_epoch_samples(
                G, loli_val_loader,
                out_dir=paths["samples_dir"], epoch=epoch,
                device=device, num_samples=4,
            )
            print(f"    → samples saved to {paths['samples_dir']}")

        # ---- CSV ----
        csv_logger.append({
            "run_tag": args.run_tag, "epoch": epoch,
            "lr": opt_g.param_groups[0]["lr"],
            "lambda_det_effective": lam_eff,
            "L_total":    sums["total"] / max(n, 1),
            "L_rec":      sums["rec"]   / max(n, 1),
            "L_l1":       sums["l1"]    / max(n, 1),
            "L_vgg":      sums["vgg"]   / max(n, 1),
            "L_ssim":     sums["ssim"]  / max(n, 1),
            "L_feat_mse": sums["feat_mse"] / max(n, 1),
            "L_feat_cos": sums["feat_cos"] / max(n, 1),
            "L_preserve": sums["preserve"] / max(n, 1),
            "L_mse_p3":   sums["mse_p3"] / max(n, 1),
            "L_mse_p4":   sums["mse_p4"] / max(n, 1),
            "L_mse_p5":   sums["mse_p5"] / max(n, 1),
            "L_cos_p3":   sums["cos_p3"] / max(n, 1),
            "L_cos_p4":   sums["cos_p4"] / max(n, 1),
            "L_cos_p5":   sums["cos_p5"] / max(n, 1),
            "train_psnr": sums["train_psnr"] / max(n, 1),
            "loli_val_psnr": loli_eval["psnr"],
            "loli_val_ssim": loli_eval["ssim"],
            "lol_eval15_psnr": lol15_eval["psnr"],
            "lol_eval15_ssim": lol15_eval["ssim"],
            "delta_eval15_psnr": (
                lol15_eval["psnr"] - baseline_eval15_psnr
                if baseline_eval15_psnr is not None else 0.0
            ),
            "exdark_map50": exdark_result["map50"] if exdark_result else "",
            "exdark_map":   exdark_result["map"]   if exdark_result else "",
            "exdark_p":     exdark_result["p"]     if exdark_result else "",
            "exdark_r":     exdark_result["r"]     if exdark_result else "",
            "best_psnr": best_psnr,
            "best_map":  best_map if best_map > -1e9 else "",
            "grad_norm_mean": sums["grad_norm"] / max(n, 1),
            "epoch_sec": epoch_sec,
        })

        # ---- _last 갱신 ----
        _save_ckpt(paths["last"], epoch, G, opt_g, sch_g,
                   best_psnr, best_map, args)

    print()
    print(HRULE)
    print(f"  완료 ✓   best_psnr = {best_psnr:.3f}   best_map@0.5 = "
          f"{(best_map if best_map > -1e9 else float('nan')):.4f}")
    print(HRULE)


# ===========================================================================
# 7. CLI
# ===========================================================================
def _run_tag(lambda_det: float, use_preserve: bool) -> str:
    """0.01 → '001', 0.05 → '005', 0.05 + preserve → '005_preserve'."""
    base = f"{int(round(lambda_det * 100)):03d}"
    return f"{base}_preserve" if use_preserve else base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LUNA detection-aware fine-tuning "
                    "(frozen YOLOv8n P3/P4/P5 feature loss, Stage 1 only)",
    )
    # ---- 데이터 ----
    p.add_argument("--loli_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LoLI-Street")
    p.add_argument("--lol_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LOL-v2")
    p.add_argument("--eval_lol_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LOLdataset",
                   help="LOL eval15 폴더 — LOL eval15 평가용.")
    p.add_argument("--exdark_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\ExDark",
                   help="ExDark 루트 (images/, annotations/).  ''(빈값) 이면 ExDark eval 비활성.")
    p.add_argument("--max_samples", type=int, default=5000,
                   help="혼합 학습셋 총 크기 (LoLI + LOL-v2).  기본 5000.")
    p.add_argument("--mix_ratio", type=float, default=0.2,
                   help="LOL-v2 비율 ∈ [0, 1].  기본 0.2 → 4k LoLI + 1k LOL-v2.")
    p.add_argument("--val_subset_size", type=int, default=500)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4,
                   help="기본 4 — 8GB VRAM 에서 YOLO feature extraction 포함 시 OOM 방지.")

    # ---- 학습 ----
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr",     type=float, default=1e-5)
    p.add_argument("--lambda_det",   type=float, default=0.01,
                   help="Detection-aware loss 의 target 가중치. warmup 후 도달.")
    p.add_argument("--alpha_cos",    type=float, default=0.5,
                   help="L_feat_cos 가중치 (DetectionAwareLoss 내부).")
    p.add_argument("--beta_preserve",type=float, default=0.1,
                   help="L_preserve 가중치 (use_preserve 시).")
    p.add_argument("--use_preserve", action="store_true",
                   help="Feature preservation loss 사용 (기본 off).")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="λ_det 가 0 → target 으로 선형 증가하는 epoch 수.")
    p.add_argument("--grad_clip_max", type=float, default=1.0,
                   help="generator gradient L2-norm clip (frozen YOLO grad 폭주 방지).")
    p.add_argument("--lambda_l1",         type=float, default=1.0)
    p.add_argument("--lambda_perceptual", type=float, default=0.5)
    p.add_argument("--lambda_ssim",       type=float, default=1.0)
    p.add_argument("--beta1",  type=float, default=0.9)
    p.add_argument("--beta2",  type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay.")
    p.add_argument("--eta_min_ratio", type=float, default=0.1)

    # ---- YOLO ----
    p.add_argument("--yolo_weights", type=str, default="yolov8n.pt",
                   help="없으면 ultralytics 가 자동 다운로드.")
    p.add_argument("--debug_yolo_layers", action="store_true",
                   help="YOLO 전체 layer index/type/shape + P3/P4/P5 선택을 출력.")

    # ---- 평가 / 모니터링 ----
    p.add_argument("--eval_every",        type=int, default=1)
    p.add_argument("--eval_every_exdark", type=int, default=5,
                   help="ExDark quick eval 주기 (epoch). 0 이면 마지막 epoch 만.")
    p.add_argument("--exdark_num_samples", type=int, default=100)
    p.add_argument("--exdark_conf",       type=float, default=0.25)
    p.add_argument("--save_samples_every", type=int, default=5)
    p.add_argument("--forget_threshold",  type=float, default=19.0)

    # ---- I/O ----
    p.add_argument("--save_dir",    type=str, default="./checkpoints")
    p.add_argument("--log_path",    type=str, default="./logs/det_aware_training.csv")
    p.add_argument("--samples_dir", type=str, default="./results/det_aware_samples")
    p.add_argument("--resume", type=str,
                   default="checkpoints/loli_street_30000_stage2_best.pth",
                   help="시작 시점의 G 가중치 (hybrid_v1 사양).")
    p.add_argument("--force", action="store_true",
                   help="해당 tag 의 기존 best/last 삭제 후 처음부터.")

    # ---- 런타임 ----
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not 0.0 <= args.mix_ratio <= 1.0:
        raise ValueError(f"--mix_ratio ∈ [0, 1] 필요 (got {args.mix_ratio})")
    if args.lambda_det < 0:
        raise ValueError(f"--lambda_det ≥ 0 필요 (got {args.lambda_det})")
    if args.warmup_epochs < 0:
        raise ValueError(f"--warmup_epochs ≥ 0 필요 (got {args.warmup_epochs})")
    args.run_tag = _run_tag(args.lambda_det, args.use_preserve)
    return args


# ===========================================================================
# 8. Main
# ===========================================================================
def main() -> int:
    args = parse_args()

    save_dir    = Path(args.save_dir).resolve()
    log_path    = Path(args.log_path).resolve()
    samples_dir = Path(args.samples_dir).resolve() / args.run_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {
        "best_psnr":   save_dir / f"det_aware_{args.run_tag}_best_psnr.pth",
        "best_map":    save_dir / f"det_aware_{args.run_tag}_best_map.pth",
        "last":        save_dir / f"det_aware_{args.run_tag}_last.pth",
        "samples_dir": samples_dir,
    }

    if args.force:
        for k in ("best_psnr", "best_map", "last"):
            if paths[k].exists(): paths[k].unlink()
        print(f"[force] 기존 det_aware_{args.run_tag}_* 체크포인트 삭제.")

    print()
    print(HRULE)
    print(" LUNA detection-aware fine-tuning")
    print(HRULE)
    print(f"  run_tag      : {args.run_tag}  (λ_det={args.lambda_det}, "
          f"use_preserve={args.use_preserve})")
    print(f"  loli_root    : {args.loli_root}")
    print(f"  lol_root     : {args.lol_root}")
    print(f"  eval_lol_root: {args.eval_lol_root}")
    print(f"  exdark_root  : {args.exdark_root or '(disabled)'}")
    print(f"  max_samples  : {args.max_samples}  (mix_ratio={args.mix_ratio})")
    print(f"  device       : {args.device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  save_dir     : {save_dir}")
    print(f"  log_path     : {log_path}")
    print(f"  samples_dir  : {samples_dir}")
    print(SUBRULE)

    # ---- 데이터 로더 ----
    print(" [1/5] 학습 데이터 (LoLI + LOL-v2 Real 혼합)")
    train_loader, info = build_mixed_train_loader(
        loli_root=Path(args.loli_root), lol_root=Path(args.lol_root),
        max_samples=args.max_samples, mix_ratio=args.mix_ratio,
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    print(f"    LoLI: {info['n_loli']}  /  LOL-v2: {info['n_lol']}  /  total: {info['n_total']}")
    print(f"    batch={args.batch_size}, steps/epoch ≈ "
          f"{info['n_total']//max(args.batch_size, 1)}")

    print(" [2/5] 평가 로더 (LoLI val + LOL eval15)")
    loli_val_loader, lol15_loader, n_val, n_e15 = build_eval_loaders(
        loli_root=Path(args.loli_root),
        eval_lol_root=Path(args.eval_lol_root),
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_subset_size=args.val_subset_size,
        seed=args.seed,
    )
    print(f"    LoLI val={n_val}  /  LOL eval15={n_e15}")

    print(" [3/5] Resume baseline 측정 (LOL eval15)")
    baseline_psnr = _measure_baseline_eval15(
        Path(args.resume), lol15_loader, device=args.device,
    )
    if baseline_psnr is not None:
        print(f"    baseline LOL eval15 PSNR = {baseline_psnr:.3f}")
    else:
        print("    baseline 측정 안됨 — Δ(eval15) 는 0 으로 표시.")

    print(" [4/5] ExDark quick eval YOLO 추론기 준비")
    yolo_predict_model = None
    exdark_active = args.exdark_root and (Path(args.exdark_root) / "images").is_dir()
    if exdark_active:
        try:
            from ultralytics import YOLO  # type: ignore
            yolo_predict_model = YOLO(args.yolo_weights)
            print(f"    ExDark eval 활성 — yolo_predict_model 로드 완료.")
        except Exception as e:
            print(f"    [warn] ExDark eval 비활성 (YOLO 로드 실패): {e}")
            exdark_active = False
    else:
        print(f"    [info] exdark_root 없음 — best_map 체크포인트는 저장되지 않습니다.")

    csv_logger = DetCsvLogger(log_path)
    print(" [5/5] 학습 시작")

    overall_t0 = time.perf_counter()
    train_detection_aware(
        args=args, paths=paths,
        train_loader=train_loader,
        loli_val_loader=loli_val_loader,
        lol15_loader=lol15_loader,
        baseline_eval15_psnr=baseline_psnr,
        csv_logger=csv_logger,
        yolo_predict_model=yolo_predict_model,
    )

    total_min = (time.perf_counter() - overall_t0) / 60.0
    print()
    print(HRULE)
    print(f"  Wall time     : {total_min:.1f} min")
    print(f"  best_psnr ckpt: {paths['best_psnr']}")
    print(f"  best_map  ckpt: {paths['best_map']}")
    print(f"  last      ckpt: {paths['last']}")
    print(f"  CSV log       : {log_path}")
    print(f"  samples       : {samples_dir}")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
