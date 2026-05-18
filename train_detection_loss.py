"""LUNA + frozen YOLOv8n direct-detection loss fine-tuning.

핵심 아이디어 (vs. train_detection_aware.py)
--------------------------------------------
* 이전 (feature matching): LUNA 출력의 YOLO feature 를 *GT 이미지* feature 와
  비슷하게 → reconstruction loss 와 학습 신호가 겹쳐 ExDark mAP 향상 효과
  미미.
* **이번 (direct detection loss)**: ExDark 의 실제 GT bbox annotation 을
  사용하여 "YOLO 가 물체를 검출하게" 라는 task 자체를 LUNA 에 역전파.
  reconstruction 과 *직교* 한 학습 신호 → mAP 를 직접 끌어올린다.

학습 흐름 (per step)
--------------------
1. *Reconstruction branch* — LoLI-Street paired (low, high) 로
        L_rec = L1 + 0.5·VGG + 1.0·SSIM   (SupervisedLoss, 이미지 품질 유지)

2. *Detection branch* — ExDark (image, YOLO-format targets) 로
        enh = G(ex_img); enh_01 = clamp((enh + 1) / 2, 0, 1)
        preds = frozen_YOLOv8n(enh_01)
        L_det, items = v8DetectionLoss(preds, targets)
        # items = (box, cls, dfl) 의 (3,) 텐서, L_det = items.sum()*batch  (단, 본 구현은
        # items 가 이미 batch 가중을 포함하므로 ``L_det = loss.sum()`` 로 처리)

3. 합산 + backward + grad clip + optimizer.step
        L_total = L_rec + λ_det_eff · L_det
        L_total.backward()
        clip_grad_norm_(G.parameters(), 1.0)

Frozen YOLO 설정 (절대 update 금지)
----------------------------------
* ``yolo_model.parameters()`` 모두 ``requires_grad=False``.
* ``yolo_model.eval()`` 호출 후 그대로 유지 (BN running stats 동결).
* 다만 ``yolo_model.model[-1].training = True`` 로 Detect head 만 "training"
  상태로 두어 raw multi-scale predictions (dict[boxes/scores/feats]) 를
  반환하도록 강제 — v8DetectionLoss 가 받는 포맷.
* gradient 는 LUNA → enh → YOLO Conv/BN → loss 로 *흐른다* (no_grad 금지).

Warmup
------
``--warmup_epochs`` (기본 3) 동안 λ_det_eff = 0 으로 reconstruction 만 학습 →
이후 선형 증가하여 ``--lambda_det`` 도달.  detection gradient 가 초기에 LUNA
를 흔드는 것 방지.

평가
----
* 매 epoch: LoLI val 500 PSNR/SSIM + LOL eval15 PSNR/SSIM.
* 매 ``--eval_every_exdark`` (기본 5) epoch: ExDark Test split N 장 (기본 100)
  YOLO inference 후 ``DetectionAccumulator`` 로 mAP@0.5 / mAP / P / R 측정.
* best_psnr / best_map 별도 저장 — 본 실험의 *주된* 지표는 best_map.

ExDark split 처리 (주의)
-----------------------
기본값: training 에 splits=(1, 2, 3) 을 모두 사용 (사용자 요청).  ExDark
quick eval 은 split=(3,) 의 100 장을 사용하므로 **train/eval overlap** 이
존재함 — 결과 해석 시 유의.  ``--exdark_train_splits "1,2"`` 로 leakage 제거 가능.

체크포인트
----------
* ``checkpoints/det_loss_{tag}_best_psnr.pth``
* ``checkpoints/det_loss_{tag}_best_map.pth``
* ``checkpoints/det_loss_{tag}_last.pth``
  ``tag = round(λ_det × 10) zero-pad 2`` → 0.1 → "01", 0.5 → "05".
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Windows 콘솔 UTF-8
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
from utils.exdark_detection_dataset import (
    ExDarkDetectionDataset, exdark_yolo_collate, move_target_dict,
)

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
# 1. Frozen YOLOv8n setup — training=True 만 Detect head 에 강제
# ===========================================================================
def setup_frozen_yolo(weights: str, device: str) -> Tuple[nn.Module, Any, Any]:
    """frozen YOLOv8n 모델 + v8DetectionLoss + ultralytics YOLO wrapper 반환.

    Returns
    -------
    (yolo_model, det_loss_fn, yolo_predict_model)
        * yolo_model       : nn.Module, eval+frozen, Detect head training=True
          (학습 step 에서 detection gradient 계산용).
        * det_loss_fn      : ``v8DetectionLoss`` instance.
        * yolo_predict_model : ultralytics ``YOLO`` wrapper (ExDark quick eval
          에서 predict() API 용 — 별도 instance 로 inference 전용).
    """
    try:
        from ultralytics import YOLO
        from ultralytics.utils import DEFAULT_CFG
        from ultralytics.utils.loss import v8DetectionLoss
    except ImportError as e:
        raise ImportError("ultralytics 필요 — pip install ultralytics") from e

    # ---- (A) detection loss 용 — params frozen, Detect head training=True ----
    yolo_inner = YOLO(weights)
    m: nn.Module = yolo_inner.model.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False

    # Detect head 강제 training=True (raw 3-scale preds 반환)
    detect_head = m.model[-1]  # type: ignore[index]
    if type(detect_head).__name__ != "Detect":
        raise RuntimeError(
            f"YOLO 마지막 module 이 Detect 가 아님: {type(detect_head).__name__}"
        )
    detect_head.training = True

    # v8DetectionLoss 가 ``model.args.box/cls/dfl`` 를 *attribute* 로 읽으므로
    # 만약 ``args`` 가 plain dict 면 IterableSimpleNamespace 로 교체.
    if not hasattr(m, "args") or m.args is None or isinstance(m.args, dict):
        m.args = DEFAULT_CFG

    det_loss_fn = v8DetectionLoss(m)

    # ---- (B) ExDark quick eval 용 — 별도 YOLO instance (predict API) ----
    yolo_predict_model = YOLO(weights)

    return m, det_loss_fn, yolo_predict_model


# ===========================================================================
# 2. λ_det warmup — 0 → target 선형 증가
# ===========================================================================
def lambda_det_for_epoch(epoch: int, target: float, warmup_epochs: int) -> float:
    """epoch 1-indexed.  W = warmup_epochs.

    구간:
        epoch ≤ W       → 0
        W < epoch ≤ 2W  → target · (epoch - W) / W   (선형 증가)
        epoch > 2W      → target
    """
    if warmup_epochs <= 0:
        return float(target)
    if epoch <= warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / float(warmup_epochs)
    return float(target) * min(progress, 1.0)


# ===========================================================================
# 3. CSV 로거
# ===========================================================================
CSV_FIELDS: List[str] = [
    "run_tag", "epoch", "lr", "lambda_det_effective",
    "L_total", "L_rec", "L_l1", "L_vgg", "L_ssim",
    "L_det", "L_det_box", "L_det_cls", "L_det_dfl",
    "train_psnr",
    "loli_val_psnr", "loli_val_ssim",
    "lol_eval15_psnr", "lol_eval15_ssim",
    "delta_eval15_psnr",
    "exdark_map50", "exdark_map", "exdark_p", "exdark_r",
    "best_psnr", "best_map",
    "grad_norm_mean", "epoch_sec",
]


class DetLossCsvLogger:
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
# 4. 체크포인트 helper
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
        "dataset":       "loli_street_mix + exdark_detection",
        "lambda_det":    float(args.lambda_det),
        "warmup_epochs": int(args.warmup_epochs),
    }
    if extra:
        state.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ===========================================================================
# 5. Sample image — ExDark low + LUNA enhanced 비교
# ===========================================================================
def save_exdark_samples(
    G: nn.Module,
    exdark_loader: DataLoader,
    out_dir: Path, epoch: int,
    device: str, num_samples: int = 4,
) -> None:
    """ExDark loader 앞쪽 N 개로 [low | enhanced] 비교 PNG 저장.

    파일명: ``out_dir / epoch_XXX_sample_NNN.png``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    G.eval()
    saved = 0
    with torch.no_grad():
        for imgs, _targets in exdark_loader:
            imgs = imgs.to(device, non_blocking=True)
            enh = G(imgs)
            for i in range(imgs.size(0)):
                if saved >= num_samples:
                    break
                out_path = out_dir / f"epoch_{epoch:03d}_sample_{saved+1:03d}.png"
                save_comparison_grid(
                    [imgs[i].detach(), enh[i].detach()],
                    out_path, ncols=2, pad=4,
                )
                saved += 1
            if saved >= num_samples:
                break


# ===========================================================================
# 6. ExDark quick mAP eval — 기존 downstream_exdark 인프라 재사용
# ===========================================================================
def quick_exdark_eval(
    G: nn.Module,
    exdark_root: Path,
    yolo_predict_model: Any,
    image_size: int, device: str,
    num_samples: int = 100, conf: float = 0.25,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    """ExDark Test split 에서 N 장 sampling → mAP/P/R 빠른 측정.  실패 시 None."""
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
    if 0 < num_samples < len(samples):
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
                gt_boxes = np.array(
                    [[x1, y1, x2, y2] for (_c, x1, y1, x2, y2) in recs],
                    dtype=np.float32,
                )
                gt_classes = np.array([c for (c, *_r) in recs], dtype=np.int64)
            else:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
                gt_classes = np.zeros((0,), dtype=np.int64)
            enh_rgb = enhance_with_luna(G, pil, image_size, device)
            r = _run_yolo(yolo_predict_model, enh_rgb, conf, device)
            pb, pc, pf = _extract_boxes(r)
            acc.add(img_id, gt_boxes, gt_classes, pb, pc, pf)

    res = acc.compute()
    return {
        "map50":   float(res["map50"]),
        "map":     float(res["map"]),
        "p":       float(res["p50"]),
        "r":       float(res["r50"]),
        "n_images": int(res["n_images"]),
    }


# ===========================================================================
# 7. Training loop
# ===========================================================================
def train_loop(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    G: nn.Module,
    yolo_model: nn.Module,
    det_loss_fn: Any,
    yolo_predict_model: Any,
    loli_loader: DataLoader,
    exdark_loader: DataLoader,
    loli_val_loader: DataLoader,
    lol15_loader: DataLoader,
    baseline_eval15_psnr: Optional[float],
    csv_logger: DetLossCsvLogger,
) -> None:
    device = args.device
    set_seed(args.seed)

    # Reconstruction loss
    rec_loss = SupervisedLoss(
        lambda_l1=args.lambda_l1,
        lambda_vgg=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim,
    ).to(device)

    opt_g = AdamW(G.parameters(), lr=args.lr,
                  betas=(args.beta1, args.beta2),
                  weight_decay=args.weight_decay)
    sch_g = CosineAnnealingLR(opt_g, T_max=args.epochs,
                              eta_min=args.lr * args.eta_min_ratio)

    # ---- Resume external G (only if no _last) ----
    if args.resume and not paths["last"].is_file():
        rp = Path(args.resume)
        if rp.is_file():
            s = _load_g_weights(G, rp, device=device)
            print(f"  resume G from    : {rp.name}  "
                  f"(prev best LoLI PSNR = {s.get('best_loli_val_psnr', float('nan'))})")
        else:
            print(f"  [warn] --resume 파일 없음: {rp}")

    # ---- Resume _last ----
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
            print("  [info] 이미 완료 — 학습 건너뜀."); return

    # 초기 평가 — best_psnr 기준점
    if start_epoch == 1 and best_psnr == -float("inf"):
        init = evaluate(G, loli_val_loader, device=device)
        best_psnr = float(init["psnr"])
        print(f"  init LoLI PSNR   : {best_psnr:.3f}")
        _save_ckpt(paths["best_psnr"], 0, G, opt_g, sch_g,
                   best_psnr, best_map, args)

    # Step 수 — 두 로더 중 짧은 쪽 길이.  짧은 쪽이 다 도는 게 자연스럽고
    # 다른 쪽은 cycle 로 무한 공급.
    steps_per_epoch = min(len(loli_loader), len(exdark_loader))
    print(f"  steps/epoch      : {steps_per_epoch}  "
          f"(loli={len(loli_loader)}, exdark={len(exdark_loader)})")
    print(SUBRULE)
    print(" 학습 시작")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        lam_eff = lambda_det_for_epoch(epoch, args.lambda_det, args.warmup_epochs)

        G.train()
        # 안전: YOLO 는 매 epoch 시작 시점에 다시 eval + Detect.training=True 재확인
        yolo_model.eval()
        yolo_model.model[-1].training = True  # type: ignore[index]

        sums = {
            "total": 0.0, "rec": 0.0, "l1": 0.0, "vgg": 0.0, "ssim": 0.0,
            "det": 0.0, "det_box": 0.0, "det_cls": 0.0, "det_dfl": 0.0,
            "train_psnr": 0.0, "grad_norm": 0.0,
        }
        n_steps = 0
        loli_iter   = iter(loli_loader)
        exdark_iter = cycle(exdark_loader)  # 길면 cycle 이 무한 공급

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"DLoss Ep {epoch:3d}/{args.epochs} λ={lam_eff:.4f}",
            ncols=140, leave=False,
        )
        for _ in pbar:
            # ---- Reconstruction batch (LoLI-Street) ----
            try:
                low, high = next(loli_iter)
            except StopIteration:
                loli_iter = iter(loli_loader)
                low, high = next(loli_iter)
            low  = low.to(device,  non_blocking=True)
            high = high.to(device, non_blocking=True)

            opt_g.zero_grad(set_to_none=True)
            enh_rec = G(low)
            rec_out = rec_loss(enh_rec, high)
            l_rec = rec_out["total"]
            total = l_rec

            # ---- Detection batch (ExDark, only if λ_eff > 0) ----
            l_det_scalar = torch.zeros((), device=device)
            det_items = torch.zeros(3, device=device)
            if lam_eff > 0.0:
                ex_img, batch_targets = next(exdark_iter)
                ex_img = ex_img.to(device, non_blocking=True)
                batch_targets = move_target_dict(batch_targets, device)

                enh_det = G(ex_img)
                enh_det_01 = ((enh_det + 1.0) * 0.5).clamp(0.0, 1.0)
                batch_targets["img"] = enh_det_01  # v8DetectionLoss 가 shape 참조

                preds = yolo_model(enh_det_01)
                loss_vec, det_items = det_loss_fn(preds, batch_targets)
                # loss_vec = (box, cls, dfl) (3,) tensor, 각 항 이미 batch 스케일 가중됨
                l_det_scalar = loss_vec.sum()
                total = l_rec + lam_eff * l_det_scalar

            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                G.parameters(), max_norm=args.grad_clip_max,
            )
            opt_g.step()

            with torch.no_grad():
                tp = psnr_metric(enh_rec.detach(), high)
            sums["total"]      += float(total.detach())
            sums["rec"]        += float(l_rec.detach())
            sums["l1"]         += float(rec_out["l1"])
            sums["vgg"]        += float(rec_out["vgg"])
            sums["ssim"]       += float(rec_out["ssim"])
            sums["det"]        += float(l_det_scalar.detach())
            sums["det_box"]    += float(det_items[0])
            sums["det_cls"]    += float(det_items[1])
            sums["det_dfl"]    += float(det_items[2])
            sums["train_psnr"] += float(tp)
            sums["grad_norm"]  += float(grad_norm)
            n_steps += 1

            pbar.set_postfix({
                "L":    f"{float(total):.3f}",
                "rec":  f"{float(l_rec):.3f}",
                "det":  f"{float(l_det_scalar):.3f}",
                "box":  f"{float(det_items[0]):.2f}",
                "cls":  f"{float(det_items[1]):.2f}",
                "PSNR": f"{float(tp):.1f}",
                "|g|":  f"{float(grad_norm):.2f}",
            })
        pbar.close()
        sch_g.step()
        epoch_sec = time.perf_counter() - t0

        # ---- 평가: LoLI val + LOL eval15 ----
        loli_eval  = evaluate(G, loli_val_loader, device=device)
        lol15_eval = evaluate(G, lol15_loader,   device=device)

        # ---- ExDark quick eval (every N or final) ----
        ex_result: Optional[Dict[str, float]] = None
        if args.exdark_root and (
            epoch % max(args.eval_every_exdark, 1) == 0 or epoch == args.epochs
        ):
            print(f"  [ep {epoch:3d}] ExDark quick eval "
                  f"({args.exdark_eval_samples} samples) ...")
            ex_result = quick_exdark_eval(
                G=G, exdark_root=Path(args.exdark_root),
                yolo_predict_model=yolo_predict_model,
                image_size=args.image_size, device=device,
                num_samples=args.exdark_eval_samples,
                conf=args.exdark_conf, seed=args.seed,
            )

        # ---- 콘솔 요약 ----
        parts = [
            f"Ep {epoch:3d}/{args.epochs}",
            f"λ_eff={lam_eff:.4f}",
            f"LoLI={loli_eval['psnr']:6.3f}/{loli_eval['ssim']:.4f}",
            f"e15={lol15_eval['psnr']:6.3f}/{lol15_eval['ssim']:.4f}",
        ]
        if baseline_eval15_psnr is not None:
            parts.append(f"Δe15={lol15_eval['psnr']-baseline_eval15_psnr:+.3f}")
        if ex_result is not None:
            parts.append(f"ExDark mAP@0.5={ex_result['map50']:.4f} "
                         f"P/R={ex_result['p']:.3f}/{ex_result['r']:.3f}")
        parts.append(f"sec={epoch_sec:.1f}")
        print("  " + " | ".join(parts))
        if baseline_eval15_psnr is not None and lol15_eval["psnr"] < args.forget_threshold:
            print(f"    ⚠️  eval15 < {args.forget_threshold} (forgetting 위험)")

        # ---- best 갱신 ----
        if loli_eval["psnr"] > best_psnr:
            best_psnr = loli_eval["psnr"]
            _save_ckpt(paths["best_psnr"], epoch, G, opt_g, sch_g,
                       best_psnr, best_map, args)
            print(f"    → new best_psnr = {best_psnr:.3f} "
                  f"(saved {paths['best_psnr'].name})")
        if ex_result is not None and ex_result["map50"] > best_map:
            best_map = float(ex_result["map50"])
            _save_ckpt(paths["best_map"], epoch, G, opt_g, sch_g,
                       best_psnr, best_map, args,
                       extra={"exdark_eval": ex_result})
            print(f"    → new best_map  = {best_map:.4f} "
                  f"(saved {paths['best_map'].name})")

        # ---- 샘플 이미지 ----
        if epoch % max(args.save_samples_every, 1) == 0 or epoch == args.epochs:
            save_exdark_samples(
                G, exdark_loader, paths["samples_dir"], epoch, device,
                num_samples=4,
            )
            print(f"    → samples saved to {paths['samples_dir']}")

        # ---- CSV ----
        denom = max(n_steps, 1)
        csv_logger.append({
            "run_tag": args.run_tag, "epoch": epoch,
            "lr":  opt_g.param_groups[0]["lr"],
            "lambda_det_effective": lam_eff,
            "L_total":   sums["total"] / denom,
            "L_rec":     sums["rec"]   / denom,
            "L_l1":      sums["l1"]    / denom,
            "L_vgg":     sums["vgg"]   / denom,
            "L_ssim":    sums["ssim"]  / denom,
            "L_det":     sums["det"]   / denom,
            "L_det_box": sums["det_box"] / denom,
            "L_det_cls": sums["det_cls"] / denom,
            "L_det_dfl": sums["det_dfl"] / denom,
            "train_psnr": sums["train_psnr"] / denom,
            "loli_val_psnr": loli_eval["psnr"],
            "loli_val_ssim": loli_eval["ssim"],
            "lol_eval15_psnr": lol15_eval["psnr"],
            "lol_eval15_ssim": lol15_eval["ssim"],
            "delta_eval15_psnr": (
                lol15_eval["psnr"] - baseline_eval15_psnr
                if baseline_eval15_psnr is not None else 0.0
            ),
            "exdark_map50": ex_result["map50"] if ex_result else "",
            "exdark_map":   ex_result["map"]   if ex_result else "",
            "exdark_p":     ex_result["p"]     if ex_result else "",
            "exdark_r":     ex_result["r"]     if ex_result else "",
            "best_psnr":    best_psnr,
            "best_map":     best_map if best_map > -1e9 else "",
            "grad_norm_mean": sums["grad_norm"] / denom,
            "epoch_sec":      epoch_sec,
        })

        _save_ckpt(paths["last"], epoch, G, opt_g, sch_g,
                   best_psnr, best_map, args)

    print()
    print(HRULE)
    print(f"  완료 ✓   best_psnr = {best_psnr:.3f}   best_map@0.5 = "
          f"{(best_map if best_map > -1e9 else float('nan')):.4f}")
    print(HRULE)


# ===========================================================================
# 8. CLI
# ===========================================================================
def _run_tag(lambda_det: float) -> str:
    """0.1 → '01', 0.5 → '05', 1.0 → '10'."""
    return f"{int(round(lambda_det * 10)):02d}"


def _parse_splits(spec: str) -> Optional[Tuple[int, ...]]:
    """``"1,2,3"`` → ``(1, 2, 3)``;  빈 문자열 or "all" → None (필터 끔)."""
    spec = spec.strip().lower()
    if not spec or spec == "all":
        return None
    out = tuple(int(s) for s in spec.split(",") if s.strip())
    return out if out else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LUNA + frozen YOLOv8n direct detection loss fine-tuning",
    )
    # ---- 데이터 ----
    p.add_argument("--loli_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LoLI-Street")
    p.add_argument("--lol_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LOL-v2")
    p.add_argument("--eval_lol_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LOLdataset")
    p.add_argument("--exdark_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\ExDark",
                   help="ExDark 루트 (images/, annotations/).  빈 문자열이면 비활성.")
    p.add_argument("--max_samples", type=int, default=5000,
                   help="LoLI + LOL-v2 mixed 학습셋 크기 (reconstruction branch).")
    p.add_argument("--mix_ratio", type=float, default=0.2)
    p.add_argument("--exdark_max_samples", type=int, default=7363,
                   help="ExDark 학습셋 크기 (detection branch).  기본 = 전체.")
    p.add_argument("--exdark_train_splits", type=str, default="1,2,3",
                   help="ExDark 학습 split.  '1,2' / '1,2,3' / 'all' 가능. "
                        "기본 1,2,3 (사용자 요청).  Eval(=split 3) 과 overlap 주의.")
    p.add_argument("--val_subset_size", type=int, default=500)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2,
                   help="기본 2 — YOLO detection loss 의 VRAM 부담으로 8GB 제한.")

    # ---- 학습 ----
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr",     type=float, default=5e-6,
                   help="detection loss 가 강하므로 baseline lr=1e-5 보다 낮춤.")
    p.add_argument("--lambda_det",      type=float, default=0.1,
                   help="Detection loss 목표 가중치 (warmup 후 도달).")
    p.add_argument("--warmup_epochs",   type=int, default=3,
                   help="λ_det 가 0 → target 으로 선형 증가하는 epoch 수.")
    p.add_argument("--grad_clip_max",   type=float, default=1.0)
    p.add_argument("--lambda_l1",         type=float, default=1.0)
    p.add_argument("--lambda_perceptual", type=float, default=0.5)
    p.add_argument("--lambda_ssim",       type=float, default=1.0)
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eta_min_ratio", type=float, default=0.1)

    # ---- YOLO ----
    p.add_argument("--yolo_weights", type=str, default="yolov8n.pt")

    # ---- 평가 ----
    p.add_argument("--eval_every",         type=int, default=1)
    p.add_argument("--eval_every_exdark",  type=int, default=5)
    p.add_argument("--exdark_eval_samples", type=int, default=100)
    p.add_argument("--exdark_conf",        type=float, default=0.25)
    p.add_argument("--save_samples_every", type=int, default=5)
    p.add_argument("--forget_threshold",   type=float, default=19.0)

    # ---- I/O ----
    p.add_argument("--save_dir",    type=str, default="./checkpoints")
    p.add_argument("--log_path",    type=str, default="./logs/det_loss_training.csv")
    p.add_argument("--samples_dir", type=str, default="./results/det_loss_samples")
    p.add_argument("--resume", type=str,
                   default="checkpoints/loli_street_30000_stage2_best.pth")
    p.add_argument("--force", action="store_true",
                   help="해당 tag 의 best/last 삭제 후 처음부터.")

    # ---- 런타임 ----
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed",   type=int, default=42)

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.lambda_det < 0:
        raise ValueError(f"--lambda_det ≥ 0 (got {args.lambda_det})")
    if args.warmup_epochs < 0:
        raise ValueError(f"--warmup_epochs ≥ 0 (got {args.warmup_epochs})")
    args.run_tag = _run_tag(args.lambda_det)
    return args


# ===========================================================================
# 9. Main
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
        "best_psnr":   save_dir / f"det_loss_{args.run_tag}_best_psnr.pth",
        "best_map":    save_dir / f"det_loss_{args.run_tag}_best_map.pth",
        "last":        save_dir / f"det_loss_{args.run_tag}_last.pth",
        "samples_dir": samples_dir,
    }
    if args.force:
        for k in ("best_psnr", "best_map", "last"):
            if paths[k].exists(): paths[k].unlink()
        print(f"[force] det_loss_{args.run_tag}_* 체크포인트 삭제.")

    print()
    print(HRULE)
    print(" LUNA + frozen YOLOv8n direct detection loss fine-tuning")
    print(HRULE)
    print(f"  run_tag         : {args.run_tag}  (λ_det={args.lambda_det})")
    print(f"  loli_root       : {args.loli_root}")
    print(f"  lol_root        : {args.lol_root}")
    print(f"  eval_lol_root   : {args.eval_lol_root}")
    print(f"  exdark_root     : {args.exdark_root}")
    print(f"  exdark_splits   : {args.exdark_train_splits}  (eval = split 3)")
    print(f"  max_samples     : {args.max_samples} (mix_ratio={args.mix_ratio})")
    print(f"  exdark_max_samp : {args.exdark_max_samples}")
    print(f"  epochs/lr       : {args.epochs} / {args.lr:.2e}  (AdamW, grad_clip={args.grad_clip_max})")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  device          : {args.device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  save_dir        : {save_dir}")
    print(f"  log_path        : {log_path}")
    print(f"  samples_dir     : {samples_dir}")
    print(SUBRULE)

    # ---- 데이터: LoLI + LOL mixed (reconstruction) ----
    print(" [1/6] LoLI + LOL-v2 Real mixed loader (reconstruction)")
    loli_loader, info = build_mixed_train_loader(
        loli_root=Path(args.loli_root), lol_root=Path(args.lol_root),
        max_samples=args.max_samples, mix_ratio=args.mix_ratio,
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    print(f"    LoLI: {info['n_loli']} / LOL-v2: {info['n_lol']} / total: {info['n_total']}")
    print(f"    batch={args.batch_size}, steps/epoch ≈ "
          f"{info['n_total']//max(args.batch_size, 1)}")

    # ---- 데이터: ExDark (detection) ----
    print(" [2/6] ExDark detection dataset (annotation-driven)")
    splits_t = _parse_splits(args.exdark_train_splits)
    if splits_t and 3 in splits_t:
        print("    [warn] training 에 split=3 (Test) 가 포함됨 — quick eval 과 overlap.")
    exdark_dataset = ExDarkDetectionDataset(
        exdark_root=Path(args.exdark_root),
        image_size=args.image_size,
        splits=splits_t,
        max_samples=args.exdark_max_samples,
        filter_empty=True,
    )
    print(f"    ExDark train samples (filter_empty=True): {len(exdark_dataset)}  "
          f"(splits = {splits_t or 'ALL'})")
    if len(exdark_dataset) == 0:
        print("[error] ExDark train sample 0 — exdark_root / splits 확인.")
        return 1
    exdark_loader = DataLoader(
        exdark_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(args.num_workers, 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=min(args.num_workers, 2) > 0,
        collate_fn=exdark_yolo_collate,
        drop_last=True,
    )

    # ---- 평가 로더 ----
    print(" [3/6] 평가 로더 (LoLI val + LOL eval15)")
    loli_val_loader, lol15_loader, n_val, n_e15 = build_eval_loaders(
        loli_root=Path(args.loli_root),
        eval_lol_root=Path(args.eval_lol_root),
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_subset_size=args.val_subset_size,
        seed=args.seed,
    )
    print(f"    LoLI val={n_val}  /  LOL eval15={n_e15}")

    # ---- Baseline 측정 ----
    print(" [4/6] Resume baseline 측정 (LOL eval15)")
    baseline_psnr = _measure_baseline_eval15(
        Path(args.resume), lol15_loader, device=args.device,
    )
    if baseline_psnr is not None:
        print(f"    baseline LOL eval15 PSNR = {baseline_psnr:.3f}")
    else:
        print("    baseline 측정 안됨.")

    # ---- 모델 ----
    print(" [5/6] 모델 (LUNA generator + frozen YOLOv8n)")
    G = build_hybrid_v1_generator().to(args.device)
    g_params = sum(p.numel() for p in G.parameters())
    print(f"    G parameters    : {g_params:,}")
    yolo_model, det_loss_fn, yolo_predict_model = setup_frozen_yolo(
        weights=args.yolo_weights, device=args.device,
    )
    y_total = sum(p.numel() for p in yolo_model.parameters())
    y_train = sum(p.numel() for p in yolo_model.parameters() if p.requires_grad)
    assert y_train == 0, "YOLO 가 frozen 이 아닙니다."
    print(f"    YOLO parameters : {y_total:,} (trainable={y_train})")
    print(f"    YOLO Detect head training-flag = {yolo_model.model[-1].training}  "  # type: ignore[index]
          f"(raw 3-scale preds 반환 OK)")

    csv_logger = DetLossCsvLogger(log_path)
    print(" [6/6] 학습 시작")

    overall_t0 = time.perf_counter()
    train_loop(
        args=args, paths=paths,
        G=G, yolo_model=yolo_model, det_loss_fn=det_loss_fn,
        yolo_predict_model=yolo_predict_model,
        loli_loader=loli_loader, exdark_loader=exdark_loader,
        loli_val_loader=loli_val_loader, lol15_loader=lol15_loader,
        baseline_eval15_psnr=baseline_psnr,
        csv_logger=csv_logger,
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
