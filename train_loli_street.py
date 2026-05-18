"""LoLI-Street domain fine-tuning — LUNA 실외 거리 야간 도메인 추가 학습.

목적 (Why)
----------
LOL v1 / LOL-v2 Real (대부분 *실내* 저조도) 로 학습한 LUNA 의 best 가중치
``checkpoints/ext_lol_v2_real_stage2_best.pth`` (LOL eval15 PSNR ≈ 20.87) 를
*실외 거리 야간* 도메인 LoLI-Street (Tanvir et al., ACCV 2024) 로 추가 학습하여
일반화 성능을 개선한다.

학습 전략 (How — GPT 조언 반영)
-------------------------------
1. **Catastrophic forgetting 방지**: LoLI-Street 만 학습하면 LOL eval15 PSNR
   이 급락하는 현상이 흔하므로, 매 epoch 의 학습 배치에 LOL-v2 Real Train 을
   ``mix_ratio`` 비율 (기본 20 %) 만큼 함께 섞는다 (rehearsal-style replay).
2. **매우 낮은 LR**: Stage 1 = 1e-5, Stage 2 = 1e-6. 기존 best 가중치를
   덮어쓰지 않고 도메인 shift 만 점진적으로 흡수.
3. **이중 평가**: 매 epoch
   * LoLI-Street val subset (기본 500 장) — 새 도메인 성능
   * LOL eval15 (15 장)                  — 기존 성능 유지 확인 (forgetting check)
   둘 다 PSNR / SSIM 출력. eval15 PSNR 이 ``--forget_threshold`` (기본 19.0) 이하면
   경고 출력.
4. **점진적 subset**: ``--max_samples`` 로 학습 데이터 크기 조절
   (1000 → 5000 → 30000). 1K 부터 시작해 회귀 없는 영역에서만 키운다.

2-Stage 파이프라인 (기존 train_hybrid_v1_final.py / train_extended.py 와 동일)
-----------------------------------------------------------------------------
* Stage 1 (Supervised fine-tuning) — 50 epochs, lr = 1e-5
    L = L1 + λ_vgg · L_vgg + λ_ssim · L_ssim
* Stage 2 (Weak adversarial fine-tuning) — 25 epochs, lr = 1e-6
    L_G = λ_adv · L_adv + L1 + λ_vgg · L_vgg + λ_ssim · L_ssim
    Discriminator + label smoothing + instance noise (기본 정책 그대로).

체크포인트 / 로그
-----------------
* ``checkpoints/loli_street_{max_samples}_stage{1,2}_{best,last}.pth``
* ``logs/loli_street_training.csv`` — epoch 별 train_loss, train_psnr,
  loli_val_psnr/ssim, lol_eval15_psnr/ssim, lr, time, forget_warning

사용 예
-------
.. code-block:: bash

    # 1K subset 으로 빠르게 1차 검증
    python train_loli_street.py --max_samples 1000 \\
        --resume checkpoints/ext_lol_v2_real_stage2_best.pth \\
        --loli_root "C:\\대학교\\Projects\\SmallSizePM_GAN_model\\DataSet\\LoLI-Street" \\
        --lol_root  "C:\\대학교\\Projects\\SmallSizePM_GAN_model\\DataSet\\LOL-v2"

    # 회귀 없으면 5K 로 확장
    python train_loli_street.py --max_samples 5000 \\
        --resume checkpoints/loli_street_1000_stage2_best.pth ...

    # 전체 (max_samples=0 → LoLI 30k 전부 + LOL-v2 비율만큼)
    python train_loli_street.py --max_samples 0 ...
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows 콘솔 UTF-8 안전 (한글 출력)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data import LOLDataset, LOLv2RealDataset, PairedImageDataset
from models import (
    CombinedLoss, DiscriminatorLoss,
    PatchGANDiscriminator, SupervisedLoss,
)
from utils import evaluate, psnr_metric
# 기존 학습 스크립트의 helper / 상수 재사용 — 동일한 hybrid_v1 사양을 보장.
from train import (
    _add_gaussian, _d_forward_fp32,
    instance_noise_std_for_epoch, set_seed,
)
from train_hybrid_v1_final import (
    HYBRID_V1_BASE_FILTERS, HYBRID_V1_CONV_CONFIG,
    HYBRID_V1_USE_ATTENTION, build_hybrid_v1_generator,
)


HRULE = "=" * 100
SUBRULE = "-" * 100


# ===========================================================================
# 1. 데이터 로더 구성 — LoLI-Street train + LOL-v2 Real train 혼합
# ===========================================================================
def _build_loli_train(
    loli_root: Path, image_size: int, augment: bool,
) -> PairedImageDataset:
    """LoLI-Street ``train/{low,high}`` 페어 데이터셋 (전체 30k).

    ``LoLIStreetDataset`` 을 직접 쓰면 ``split="train"`` 만 인식하므로 본 함수는
    ``PairedImageDataset`` 으로 폴더를 직접 지정해 미러본 구조 차이도 흡수한다.
    """
    low  = loli_root / "train" / "low"
    high = loli_root / "train" / "high"
    return PairedImageDataset(
        low_dir=low, high_dir=high,
        image_size=image_size, augment=augment,
        name="LoLI-Street[train]",
    )


def _build_loli_val(
    loli_root: Path, image_size: int,
) -> PairedImageDataset:
    """LoLI-Street ``val/{low,high}`` 페어 (3,000 장).  augment 끄기."""
    low  = loli_root / "val" / "low"
    high = loli_root / "val" / "high"
    return PairedImageDataset(
        low_dir=low, high_dir=high,
        image_size=image_size, augment=False,
        name="LoLI-Street[val]",
    )


def _seeded_indices(n_total: int, n_take: int, seed: int,
                    replace: bool) -> List[int]:
    """``n_total`` 에서 ``n_take`` 개 인덱스 추출 (재현 가능 seed).

    * replace=False 면 sample without replacement (n_take ≤ n_total 가정).
    * replace=True 면 동일 인덱스 재사용 허용 (LOL-v2 처럼 풀이 작을 때 oversample).
    """
    rng = random.Random(seed)
    if not replace:
        return rng.sample(range(n_total), k=n_take)
    return [rng.randrange(n_total) for _ in range(n_take)]


def build_mixed_train_loader(
    loli_root: Path, lol_root: Path,
    max_samples: int, mix_ratio: float,
    image_size: int, batch_size: int, num_workers: int,
    seed: int,
) -> Tuple[DataLoader, Dict[str, int]]:
    """LoLI-Street + LOL-v2 Real 의 혼합 학습 로더.

    Parameters
    ----------
    max_samples : int
        총 혼합 학습셋 크기.  0 이면 "LoLI-Street train 전체 (30k) + LOL-v2 Real
        Train 을 mix_ratio 에 맞춰 oversample" 모드.
    mix_ratio : float
        LOL-v2 Real 의 비율 ∈ [0, 1].
        예) max_samples=1000, mix_ratio=0.2 → LoLI 800 + LOL-v2 200.

    Returns
    -------
    (loader, info)
        info 는 ``{"n_loli": int, "n_lol": int, "n_total": int,
                    "loli_pool": int, "lol_pool": int}`` 디버그 출력용.
    """
    # ---- 풀 데이터셋 (augment ON, 256 resize, PairedAugment 기본 정책) ----
    loli_full = _build_loli_train(loli_root, image_size=image_size, augment=True)
    lol_full  = LOLv2RealDataset(
        data_root=lol_root, split="train",
        image_size=image_size, augment=True,
    )

    n_loli_pool = len(loli_full)
    n_lol_pool  = len(lol_full)

    # ---- 목표 개수 계산 ----
    if max_samples <= 0:
        # 전체 모드: LoLI 전체 + LOL-v2 를 비율 맞추어 oversample
        n_loli = n_loli_pool
        # n_lol / (n_lol + n_loli) = mix_ratio  →  n_lol = mix_ratio/(1-mix_ratio) * n_loli
        if mix_ratio >= 0.999:
            n_lol = n_loli * 999   # 거의 LOL 만 — 비현실적, 보호
        elif mix_ratio <= 0.0:
            n_lol = 0
        else:
            n_lol = int(round(n_loli * mix_ratio / max(1.0 - mix_ratio, 1e-6)))
    else:
        n_lol  = int(round(max_samples * mix_ratio))
        n_loli = max_samples - n_lol

    # ---- 인덱스 추출 ----
    if n_loli > n_loli_pool:
        print(f"  [warn] LoLI 요청 {n_loli} > 풀 {n_loli_pool} — replace=True 로 oversample")
    if n_lol > n_lol_pool:
        print(f"  [warn] LOL-v2 요청 {n_lol} > 풀 {n_lol_pool} — replace=True 로 oversample")

    loli_idx = _seeded_indices(n_loli_pool, n_loli, seed=seed + 1,
                               replace=(n_loli > n_loli_pool))
    lol_idx  = _seeded_indices(n_lol_pool,  n_lol,  seed=seed + 2,
                               replace=(n_lol  > n_lol_pool))

    parts: List[torch.utils.data.Dataset] = []
    if n_loli > 0: parts.append(Subset(loli_full, loli_idx))
    if n_lol  > 0: parts.append(Subset(lol_full,  lol_idx))
    if not parts:
        raise ValueError("혼합 학습셋의 크기가 0 입니다. --max_samples / --mix_ratio 확인.")
    mixed = ConcatDataset(parts)

    gen = torch.Generator(); gen.manual_seed(seed)
    loader = DataLoader(
        mixed, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=gen,
    )
    info = {
        "n_loli": n_loli, "n_lol": n_lol, "n_total": len(mixed),
        "loli_pool": n_loli_pool, "lol_pool": n_lol_pool,
    }
    return loader, info


def build_eval_loaders(
    loli_root: Path, eval_lol_root: Path,
    image_size: int, num_workers: int,
    val_subset_size: int, seed: int,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """평가 로더 두 개 + 각각의 페어 수.

    * LoLI-Street val subset (랜덤 ``val_subset_size`` 장, seed 고정 → 재현)
    * LOL eval15 (15 장)
    """
    # LoLI val
    loli_val_full = _build_loli_val(loli_root, image_size=image_size)
    if val_subset_size > 0 and val_subset_size < len(loli_val_full):
        idx = _seeded_indices(len(loli_val_full), val_subset_size,
                              seed=seed + 100, replace=False)
        loli_val = Subset(loli_val_full, idx)
    else:
        loli_val = loli_val_full

    loli_val_loader = DataLoader(
        loli_val, batch_size=1,
        num_workers=min(num_workers, 2), shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=min(num_workers, 2) > 0,
    )

    # LOL eval15 — 평가 일관성을 위해 augment=False
    eval15_root = Path(eval_lol_root)
    if not (eval15_root / "eval15" / "low").is_dir():
        raise FileNotFoundError(
            f"LOL eval15 폴더가 없습니다: {eval15_root / 'eval15'}\n"
            f"  --eval_lol_root 로 LOLdataset 경로를 명시하세요."
        )
    lol15 = LOLDataset(
        data_root=eval15_root, split="eval",
        image_size=image_size, augment=False,
    )
    lol15_loader = DataLoader(
        lol15, batch_size=1,
        num_workers=min(num_workers, 2), shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=min(num_workers, 2) > 0,
    )
    return loli_val_loader, lol15_loader, len(loli_val), len(lol15)


# ===========================================================================
# 2. CSV 로그
# ===========================================================================
CSV_FIELDS = [
    "stage", "epoch", "lr_g", "lr_d",
    "train_g_loss", "train_d_loss", "train_psnr",
    "loli_val_psnr", "loli_val_ssim",
    "lol_eval15_psnr", "lol_eval15_ssim",
    "delta_eval15_psnr",                   # baseline 대비 차이
    "forget_warning",                       # 0/1
    "best_loli_val_psnr",
    "epoch_sec", "noise_std",
]


class CsvLogger:
    """단일 CSV 파일에 stage1 + stage2 행을 모두 추가하는 단순 로거."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # 새 파일이면 header 작성. 기존 파일이면 append 모드.
        if not path.is_file() or path.stat().st_size == 0:
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        # 누락된 필드 0/빈값 보정
        full = {k: row.get(k, "") for k in CSV_FIELDS}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(full)


# ===========================================================================
# 3. 체크포인트 helper
# ===========================================================================
def _save_ckpt(
    path: Path, stage: int, epoch: int, best_loli_psnr: float,
    G: nn.Module, opt_g, sch_g, sc_g: GradScaler,
    D: Optional[nn.Module] = None,
    opt_d=None, sch_d=None, sc_d: Optional[GradScaler] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Stage 1/2 공통 체크포인트 저장. Stage 1 면 D 관련 필드는 None 으로 무시."""
    state: Dict[str, Any] = {
        "stage": stage, "epoch": epoch,
        "generator":     G.state_dict(),
        "opt_g":         opt_g.state_dict(),
        "sch_g":         sch_g.state_dict(),
        "sc_g":          sc_g.state_dict(),
        "best_loli_val_psnr": best_loli_psnr,
        "base_filters":  HYBRID_V1_BASE_FILTERS,
        "use_attention": HYBRID_V1_USE_ATTENTION,
        "conv_config":   HYBRID_V1_CONV_CONFIG.copy(),
        "dataset":       "loli_street_mix",
    }
    if D is not None:
        state.update({
            "discriminator": D.state_dict(),
            "opt_d":         opt_d.state_dict(),
            "sch_d":         sch_d.state_dict(),
            "sc_d":          sc_d.state_dict(),
        })
    if extra:
        state.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_g_weights(G: nn.Module, ckpt_path: Path, device: str) -> Dict[str, Any]:
    """체크포인트에서 generator 가중치만 G 로 로드. architecture 불일치는 에러."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["generator"] if "generator" in state else state
    try:
        G.load_state_dict(sd)
    except RuntimeError as e:
        raise RuntimeError(
            f"--resume {ckpt_path} 의 G 가중치 로드 실패 — architecture 불일치.\n"
            f"  본 스크립트는 hybrid_v1 (base_filters=32) 고정.\n"
            f"  체크포인트의 base_filters = {state.get('base_filters')}\n"
            f"  체크포인트의 conv_config  = {state.get('conv_config')}\n"
            f"  원본 에러: {e}"
        ) from e
    return state


# ===========================================================================
# 4. Forgetting 경고
# ===========================================================================
def _format_eval_line(
    epoch: int, total: int, stage: int,
    loli: Dict[str, float], lol15: Dict[str, float],
    baseline_eval15: Optional[float],
    threshold: float,
) -> Tuple[str, bool]:
    """매 epoch 평가 한 줄 요약 + forgetting 경고 여부.

    Returns
    -------
    (text, warned) : 콘솔용 텍스트, 경고가 발동되었는지.
    """
    delta = (lol15["psnr"] - baseline_eval15) if baseline_eval15 is not None else 0.0
    warned = lol15["psnr"] < threshold
    parts = [
        f"S{stage} Ep {epoch:3d}/{total}",
        f"LoLI val PSNR/SSIM = {loli['psnr']:6.3f}/{loli['ssim']:.4f}",
        f"LOL eval15 = {lol15['psnr']:6.3f}/{lol15['ssim']:.4f}",
    ]
    if baseline_eval15 is not None:
        parts.append(f"Δ(eval15) = {delta:+.3f}")
    line = "  " + " | ".join(parts)
    if warned:
        line += f"   ⚠️ eval15 < {threshold} (catastrophic forgetting 위험)"
    return line, warned


# ===========================================================================
# 5. Stage 1 — Supervised fine-tuning
# ===========================================================================
def run_stage1(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    train_loader: DataLoader,
    loli_val_loader: DataLoader,
    lol15_loader: DataLoader,
    baseline_eval15_psnr: Optional[float],
    csv_logger: CsvLogger,
) -> Path:
    """Stage 1 — L1 + VGG + SSIM, lr=1e-5, 기존 G 가중치에서 이어서."""
    device = args.device
    set_seed(args.seed)
    use_amp = (not args.no_amp) and device == "cuda"
    if use_amp and args.lr_stage1 > 2e-4:
        print(f"[stage1] WARN: lr={args.lr_stage1:.0e} + AMP → overflow 위험. AMP 자동 비활성.")
        use_amp = False

    print()
    print(HRULE)
    print(f" ★ STAGE 1 — Supervised fine-tuning  (loli_street mix)")
    print(HRULE)
    print(f"  epochs / lr    : {args.stage1_epochs} / {args.lr_stage1:.2e}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  resume         : {args.resume}")
    print(f"  baseline psnr  : {baseline_eval15_psnr}")
    print(f"  device / AMP   : {device} / {'on' if use_amp else 'off'}")
    print(SUBRULE)

    G = build_hybrid_v1_generator().to(device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"  parameters     : {n_params:,}")

    # ---- Resume from external G weights — only if no stage1_last yet ----
    if args.resume and not paths["stage1_last"].is_file():
        rp = Path(args.resume)
        if rp.is_file():
            s = _load_g_weights(G, rp, device=device)
            print(f"[stage1] resume G from {rp.name}  "
                  f"(prev best PSNR = {s.get('best_psnr', float('nan'))})")
        else:
            print(f"[stage1] [warn] --resume 파일 없음: {rp} — 무작위 초기화로 시작.")

    opt_g = Adam(G.parameters(), lr=args.lr_stage1,
                 betas=(args.beta1, args.beta2))
    sch_g = CosineAnnealingLR(opt_g, T_max=args.stage1_epochs,
                              eta_min=args.lr_stage1 * args.eta_min_ratio)
    sc_g = GradScaler(device="cuda", enabled=use_amp)
    loss_fn = SupervisedLoss(
        lambda_l1=args.lambda_l1,
        lambda_vgg=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim,
    ).to(device)

    # ---- Resume from stage1_last (epoch-by-epoch) ----
    start_epoch = 1
    best_loli_psnr: float = -float("inf")
    if paths["stage1_last"].is_file():
        s = torch.load(paths["stage1_last"], map_location=device, weights_only=False)
        G.load_state_dict(s["generator"])
        opt_g.load_state_dict(s["opt_g"])
        sch_g.load_state_dict(s["sch_g"])
        sc_g.load_state_dict(s["sc_g"])
        start_epoch = int(s.get("epoch", 0)) + 1
        best_loli_psnr = float(s.get("best_loli_val_psnr", -float("inf")))
        print(f"[stage1] resume from {paths['stage1_last'].name} → "
              f"epoch {start_epoch}, best LoLI val PSNR={best_loli_psnr:.3f}")
        if start_epoch > args.stage1_epochs:
            print(f"[stage1] 이미 완료 — 학습 건너뜀.")
            return paths["stage1_best"]

    try:
        for epoch in range(start_epoch, args.stage1_epochs + 1):
            t0 = time.perf_counter()
            G.train()
            sums = {"g_loss": 0.0, "g_l1": 0.0, "g_vgg": 0.0,
                    "g_ssim": 0.0, "train_psnr": 0.0}
            n = 0

            pbar = tqdm(train_loader,
                        desc=f"S1 Ep {epoch:3d}/{args.stage1_epochs}",
                        ncols=110, leave=False)
            for low, high in pbar:
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)
                bs = low.size(0)

                opt_g.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=use_amp):
                    fake = G(low)
                    losses = loss_fn(fake, high)
                    loss = losses["total"]
                sc_g.scale(loss).backward()
                sc_g.step(opt_g); sc_g.update()

                with torch.no_grad():
                    train_psnr = psnr_metric(fake.detach(), high)
                sums["g_loss"]     += float(loss.detach()) * bs
                sums["g_l1"]       += float(losses["l1"])  * bs
                sums["g_vgg"]      += float(losses["vgg"]) * bs
                sums["g_ssim"]     += float(losses["ssim"])* bs
                sums["train_psnr"] += train_psnr * bs
                n += bs
                pbar.set_postfix({
                    "L":    f"{float(loss):.3f}",
                    "L1":   f"{float(losses['l1']):.3f}",
                    "PSNR": f"{train_psnr:.1f}",
                })
            pbar.close()
            sch_g.step()
            epoch_sec = time.perf_counter() - t0

            # ---- 이중 평가 ----
            loli_eval  = evaluate(G, loli_val_loader, device=device)
            lol15_eval = evaluate(G, lol15_loader,   device=device)

            text, warned = _format_eval_line(
                epoch, args.stage1_epochs, 1,
                loli_eval, lol15_eval,
                baseline_eval15_psnr, args.forget_threshold,
            )
            print(text)

            # ---- best 갱신 + CSV ----
            improved = loli_eval["psnr"] > best_loli_psnr
            if improved:
                best_loli_psnr = loli_eval["psnr"]
                _save_ckpt(paths["stage1_best"], 1, epoch, best_loli_psnr,
                           G, opt_g, sch_g, sc_g)
                print(f"    -> new best LoLI val PSNR={best_loli_psnr:.3f}  "
                      f"(saved {paths['stage1_best'].name})")

            csv_logger.append({
                "stage": 1, "epoch": epoch,
                "lr_g": opt_g.param_groups[0]["lr"], "lr_d": 0.0,
                "train_g_loss": sums["g_loss"] / max(n, 1),
                "train_d_loss": 0.0,
                "train_psnr":   sums["train_psnr"] / max(n, 1),
                "loli_val_psnr": loli_eval["psnr"],
                "loli_val_ssim": loli_eval["ssim"],
                "lol_eval15_psnr": lol15_eval["psnr"],
                "lol_eval15_ssim": lol15_eval["ssim"],
                "delta_eval15_psnr":
                    (lol15_eval["psnr"] - baseline_eval15_psnr)
                    if baseline_eval15_psnr is not None else 0.0,
                "forget_warning": int(warned),
                "best_loli_val_psnr": best_loli_psnr,
                "epoch_sec": epoch_sec,
                "noise_std": 0.0,
            })

            _save_ckpt(paths["stage1_last"], 1, epoch, best_loli_psnr,
                       G, opt_g, sch_g, sc_g)
    except KeyboardInterrupt:
        print("\n[stage1] Ctrl+C — 마지막 epoch 까지 stage1_last 에 이미 저장됨.")
        raise

    print(f"[stage1] 완료. best LoLI val PSNR = {best_loli_psnr:.3f}")
    return paths["stage1_best"]


# ===========================================================================
# 6. Stage 2 — Weak adversarial fine-tuning
# ===========================================================================
def run_stage2(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    stage1_best: Path,
    train_loader: DataLoader,
    loli_val_loader: DataLoader,
    lol15_loader: DataLoader,
    baseline_eval15_psnr: Optional[float],
    csv_logger: CsvLogger,
) -> Path:
    """Stage 2 — Stage 1 의 G 로 시작, D 추가, lr=1e-6, λ_adv 작게."""
    device = args.device
    set_seed(args.seed)
    use_spectral_norm = not args.no_spectral_norm
    use_amp = (not args.no_amp) and device == "cuda"
    if use_spectral_norm and use_amp:
        print("[stage2] WARN: SN + AMP → NaN. AMP 자동 비활성.")
        use_amp = False

    print()
    print(HRULE)
    print(f" ★ STAGE 2 — Weak adversarial fine-tuning  (loli_street mix)")
    print(HRULE)
    print(f"  Stage 1 best   : {stage1_best}")
    print(f"  epochs         : {args.stage2_epochs}")
    print(f"  lr_g / lr_d    : {args.lr_g_stage2:.2e} / {args.lr_d_stage2:.2e}")
    print(f"  λ_adv          : {args.lambda_adv}")
    print(f"  SN / AMP       : {'on' if use_spectral_norm else 'off'} / "
          f"{'on' if use_amp else 'off'}")
    print(SUBRULE)

    G = build_hybrid_v1_generator().to(device)
    D = PatchGANDiscriminator(use_spectral_norm=use_spectral_norm).to(device)

    opt_g = Adam(G.parameters(), lr=args.lr_g_stage2,
                 betas=(args.beta1, args.beta2))
    opt_d = Adam(D.parameters(), lr=args.lr_d_stage2,
                 betas=(args.beta1, args.beta2))
    sch_g = CosineAnnealingLR(opt_g, T_max=args.stage2_epochs,
                              eta_min=args.lr_g_stage2 * args.eta_min_ratio)
    sch_d = CosineAnnealingLR(opt_d, T_max=args.stage2_epochs,
                              eta_min=args.lr_d_stage2 * args.eta_min_ratio)
    sc_g = GradScaler(device="cuda", enabled=use_amp)
    sc_d = GradScaler(device="cuda", enabled=use_amp)

    combined_loss = CombinedLoss(
        lambda_adv=args.lambda_adv,
        lambda_l1=args.lambda_l1,
        lambda_vgg=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim,
    ).to(device)
    d_loss_fn = DiscriminatorLoss(real_label=args.label_smoothing_real).to(device)

    # ---- Resume 우선순위: stage2_last → stage1_best ----
    start_epoch = 1
    best_loli_psnr: float = -float("inf")
    if paths["stage2_last"].is_file():
        s = torch.load(paths["stage2_last"], map_location=device, weights_only=False)
        G.load_state_dict(s["generator"]); D.load_state_dict(s["discriminator"])
        opt_g.load_state_dict(s["opt_g"]); opt_d.load_state_dict(s["opt_d"])
        sch_g.load_state_dict(s["sch_g"]); sch_d.load_state_dict(s["sch_d"])
        sc_g.load_state_dict(s["sc_g"]);   sc_d.load_state_dict(s["sc_d"])
        start_epoch = int(s.get("epoch", 0)) + 1
        best_loli_psnr = float(s.get("best_loli_val_psnr", -float("inf")))
        print(f"[stage2] resume from {paths['stage2_last'].name} → "
              f"epoch {start_epoch}, best={best_loli_psnr:.3f}")
        if start_epoch > args.stage2_epochs:
            print(f"[stage2] 이미 완료 — 학습 건너뜀.")
            return paths["stage2_best"]
    else:
        s1 = torch.load(stage1_best, map_location=device, weights_only=False)
        G.load_state_dict(s1["generator"])
        initial = evaluate(G, loli_val_loader, device=device)
        best_loli_psnr = initial["psnr"]
        print(f"[stage2] Stage 1 G 로드 완료. 시작 LoLI val PSNR = {best_loli_psnr:.3f}")
        _save_ckpt(paths["stage2_best"], 2, 0, best_loli_psnr,
                   G, opt_g, sch_g, sc_g,
                   D=D, opt_d=opt_d, sch_d=sch_d, sc_d=sc_d)

    try:
        for epoch in range(start_epoch, args.stage2_epochs + 1):
            t0 = time.perf_counter()
            noise_std = instance_noise_std_for_epoch(
                epoch, args.instance_noise_std,
                args.instance_noise_decay_epochs,
            )
            G.train(); D.train()
            sums = {"d_loss": 0.0, "g_loss": 0.0,
                    "g_adv": 0.0, "g_l1": 0.0, "g_vgg": 0.0, "g_ssim": 0.0,
                    "train_psnr": 0.0}
            n = 0; d_n = 0

            pbar = tqdm(train_loader,
                        desc=f"S2 Ep {epoch:3d}/{args.stage2_epochs}",
                        ncols=120, leave=False)
            for step, (low, high) in enumerate(pbar):
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)
                bs = low.size(0)

                with autocast(device_type="cuda", enabled=use_amp):
                    fake = G(low)

                # ---- D step (매 d_update_freq 마다) ----
                do_d_step = (step % max(args.d_update_freq, 1)) == 0
                last_d = float("nan")
                if do_d_step:
                    for p in D.parameters(): p.requires_grad = True
                    opt_d.zero_grad(set_to_none=True)
                    real_pair = _add_gaussian(
                        torch.cat([low, high], dim=1).float(), noise_std)
                    fake_pair = _add_gaussian(
                        torch.cat([low, fake.detach()], dim=1).float(), noise_std)
                    d_real = _d_forward_fp32(D, real_pair)
                    d_fake = _d_forward_fp32(D, fake_pair)
                    d_losses = d_loss_fn(d_real, d_fake)
                    loss_d = d_losses["total"]
                    sc_d.scale(loss_d).backward()
                    sc_d.step(opt_d); sc_d.update()
                    last_d = float(loss_d.detach())
                    sums["d_loss"] += last_d * bs
                    d_n += bs

                # ---- G step ----
                for p in D.parameters(): p.requires_grad = False
                opt_g.zero_grad(set_to_none=True)
                g_d_input = _add_gaussian(
                    torch.cat([low, fake], dim=1).float(), noise_std)
                d_fake_for_g = _d_forward_fp32(D, g_d_input)
                with autocast(device_type="cuda", enabled=use_amp):
                    g_losses = combined_loss(d_fake_for_g, fake, high)
                    loss_g = g_losses["total"]
                sc_g.scale(loss_g).backward()
                sc_g.step(opt_g); sc_g.update()

                with torch.no_grad():
                    train_psnr = psnr_metric(fake.detach(), high)
                sums["g_loss"]     += float(loss_g.detach()) * bs
                sums["g_adv"]      += float(g_losses["adv"]) * bs
                sums["g_l1"]       += float(g_losses["l1"]) * bs
                sums["g_vgg"]      += float(g_losses["vgg"]) * bs
                sums["g_ssim"]     += float(g_losses["ssim"]) * bs
                sums["train_psnr"] += train_psnr * bs
                n += bs
                pbar.set_postfix({
                    "D":    f"{last_d:.3f}" if do_d_step else "  -  ",
                    "G":    f"{float(loss_g):.3f}",
                    "PSNR": f"{train_psnr:.1f}",
                    "noise":f"{noise_std:.3f}",
                })
            pbar.close()
            sch_g.step(); sch_d.step()
            epoch_sec = time.perf_counter() - t0

            # ---- 평가 ----
            loli_eval  = evaluate(G, loli_val_loader, device=device)
            lol15_eval = evaluate(G, lol15_loader,   device=device)

            text, warned = _format_eval_line(
                epoch, args.stage2_epochs, 2,
                loli_eval, lol15_eval,
                baseline_eval15_psnr, args.forget_threshold,
            )
            print(text)

            improved = loli_eval["psnr"] > best_loli_psnr
            if improved:
                best_loli_psnr = loli_eval["psnr"]
                _save_ckpt(paths["stage2_best"], 2, epoch, best_loli_psnr,
                           G, opt_g, sch_g, sc_g,
                           D=D, opt_d=opt_d, sch_d=sch_d, sc_d=sc_d)
                print(f"    -> new best LoLI val PSNR={best_loli_psnr:.3f}  "
                      f"(saved {paths['stage2_best'].name})")

            csv_logger.append({
                "stage": 2, "epoch": epoch,
                "lr_g": opt_g.param_groups[0]["lr"],
                "lr_d": opt_d.param_groups[0]["lr"],
                "train_g_loss": sums["g_loss"] / max(n, 1),
                "train_d_loss": sums["d_loss"] / max(d_n, 1),
                "train_psnr":   sums["train_psnr"] / max(n, 1),
                "loli_val_psnr": loli_eval["psnr"],
                "loli_val_ssim": loli_eval["ssim"],
                "lol_eval15_psnr": lol15_eval["psnr"],
                "lol_eval15_ssim": lol15_eval["ssim"],
                "delta_eval15_psnr":
                    (lol15_eval["psnr"] - baseline_eval15_psnr)
                    if baseline_eval15_psnr is not None else 0.0,
                "forget_warning": int(warned),
                "best_loli_val_psnr": best_loli_psnr,
                "epoch_sec": epoch_sec,
                "noise_std": noise_std,
            })

            _save_ckpt(paths["stage2_last"], 2, epoch, best_loli_psnr,
                       G, opt_g, sch_g, sc_g,
                       D=D, opt_d=opt_d, sch_d=sch_d, sc_d=sc_d)
    except KeyboardInterrupt:
        print("\n[stage2] Ctrl+C — stage2_last 에 마지막 epoch 저장됨.")
        raise

    print(f"[stage2] 완료. best LoLI val PSNR = {best_loli_psnr:.3f}")
    return paths["stage2_best"]


# ===========================================================================
# 7. CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoLI-Street domain fine-tuning (Stage1 + Stage2, mix with LOL-v2 Real)",
    )

    # ---- 데이터 ----
    p.add_argument("--loli_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LoLI-Street",
                   help="LoLI-Street 데이터셋 루트 (train/, val/ 포함).")
    p.add_argument("--lol_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LOL-v2",
                   help="LOL-v2 데이터셋 루트 (Real_captured/Train/ 포함).")
    p.add_argument("--eval_lol_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LOLdataset",
                   help="LOL v1 LOLdataset 폴더 (eval15/ 포함).  기존 도메인 PSNR 추적용.")
    p.add_argument("--max_samples", type=int, default=1000,
                   help="혼합 학습셋 총 크기. 0 이면 LoLI 전체(30k) + LOL-v2 비율 oversample.")
    p.add_argument("--mix_ratio", type=float, default=0.2,
                   help="LOL-v2 Real 의 비율 ∈ [0, 1].  기본 0.2 = 20%% rehearsal.")
    p.add_argument("--val_subset_size", type=int, default=500,
                   help="매 epoch LoLI val 평가에 사용할 subset 크기 (0=전체 3000).")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)

    # ---- 학습 ----
    p.add_argument("--stage1_epochs", type=int, default=50)
    p.add_argument("--lr_stage1",    type=float, default=1e-5)
    p.add_argument("--stage2_epochs", type=int, default=25)
    p.add_argument("--lr_g_stage2",  type=float, default=1e-6)
    p.add_argument("--lr_d_stage2",  type=float, default=1e-6)
    p.add_argument("--lambda_adv",   type=float, default=0.01)
    p.add_argument("--lambda_l1",        type=float, default=1.0)
    p.add_argument("--lambda_perceptual",type=float, default=0.5)
    p.add_argument("--lambda_ssim",      type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eta_min_ratio", type=float, default=0.1,
                   help="cosine scheduler 의 최소 lr 비율 (lr * eta_min_ratio).")

    # ---- GAN 안정화 ----
    p.add_argument("--d_update_freq", type=int, default=2)
    p.add_argument("--label_smoothing_real", type=float, default=0.9)
    p.add_argument("--instance_noise_std", type=float, default=0.1)
    p.add_argument("--instance_noise_decay_epochs", type=int, default=10)
    p.add_argument("--no_spectral_norm", action="store_true")

    # ---- 회귀 모니터링 ----
    p.add_argument("--forget_threshold", type=float, default=19.0,
                   help="LOL eval15 PSNR 이 이 값 이하가 되면 경고 출력. (기본 19.0)")

    # ---- I/O ----
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--log_dir",  type=str, default="./logs")
    p.add_argument("--resume", type=str,
                   default="checkpoints/ext_lol_v2_real_stage2_best.pth",
                   help="시작 시점의 G 가중치.  hybrid_v1 사양이어야 함.")
    p.add_argument("--force_stage1", action="store_true")
    p.add_argument("--force_stage2", action="store_true")
    p.add_argument("--skip_stage2", action="store_true",
                   help="Stage 1 만 돌리고 종료 (1K subset 빠른 검증용).")
    p.add_argument("--stage2_only", action="store_true",
                   help="Stage 1 을 건너뛰고 --resume 의 G 가중치에서 Stage 2 만 "
                        "처음부터 실행. 기존 stage2_best/last 체크포인트는 삭제됨.")

    # ---- 런타임 ----
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not 0.0 <= args.mix_ratio <= 1.0:
        raise ValueError(f"--mix_ratio 는 [0, 1] 범위여야 합니다 (got {args.mix_ratio})")
    return args


# ===========================================================================
# 8. Main
# ===========================================================================
def _measure_baseline_eval15(
    resume_ckpt: Path, lol15_loader: DataLoader, device: str,
) -> Optional[float]:
    """``--resume`` 가중치의 LOL eval15 PSNR 을 한 번 측정해 baseline 으로 보관.

    매 epoch 의 ``Δ(eval15)`` 표시에 사용된다. 측정 실패 시 None.
    """
    if not resume_ckpt.is_file():
        return None
    try:
        G = build_hybrid_v1_generator().to(device)
        _load_g_weights(G, resume_ckpt, device=device)
        m = evaluate(G, lol15_loader, device=device)
        del G; torch.cuda.empty_cache() if device == "cuda" else None
        return float(m["psnr"])
    except Exception as e:
        print(f"  [warn] baseline eval15 측정 실패: {e}")
        return None


def main() -> int:
    args = parse_args()

    save_dir = Path(args.save_dir).resolve()
    log_dir  = Path(args.log_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- 체크포인트 파일명 ----
    n_tag = "all" if args.max_samples == 0 else str(args.max_samples)
    paths: Dict[str, Path] = {
        "stage1_best": save_dir / f"loli_street_{n_tag}_stage1_best.pth",
        "stage1_last": save_dir / f"loli_street_{n_tag}_stage1_last.pth",
        "stage2_best": save_dir / f"loli_street_{n_tag}_stage2_best.pth",
        "stage2_last": save_dir / f"loli_street_{n_tag}_stage2_last.pth",
    }
    csv_path = log_dir / "loli_street_training.csv"

    # ---- force 옵션 ----
    if args.force_stage1:
        for k in ("stage1_best","stage1_last","stage2_best","stage2_last"):
            if paths[k].exists(): paths[k].unlink()
        print("[force_stage1] 기존 체크포인트 삭제 후 처음부터.")
    elif args.force_stage2 or args.stage2_only:
        for k in ("stage2_best","stage2_last"):
            if paths[k].exists(): paths[k].unlink()
        tag = "stage2_only" if args.stage2_only else "force_stage2"
        print(f"[{tag}] Stage 2 체크포인트 삭제 후 재시작.")

    # ---- --stage2_only 사전 검증 ----
    if args.stage2_only:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            print(f"[error] --stage2_only 는 유효한 --resume 가 필요합니다: {resume_path}")
            return 1
        if args.skip_stage2:
            print("[error] --stage2_only 와 --skip_stage2 는 동시에 사용할 수 없습니다.")
            return 1

    print()
    print(HRULE)
    print(" LUNA LoLI-Street domain fine-tuning")
    print(HRULE)
    print(f"  loli_root    : {args.loli_root}")
    print(f"  lol_root     : {args.lol_root}")
    print(f"  eval_lol_root: {args.eval_lol_root}")
    print(f"  max_samples  : {args.max_samples}  (mix_ratio = {args.mix_ratio})")
    print(f"  val_subset   : {args.val_subset_size} (LoLI val)")
    print(f"  device       : {args.device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  save_dir     : {save_dir}")
    print(f"  log csv      : {csv_path}")
    print(SUBRULE)

    # ---- 데이터 로더 ----
    print(" [1/4] 학습 데이터 로더 준비 (LoLI + LOL-v2 Real 혼합)")
    train_loader, info = build_mixed_train_loader(
        loli_root=Path(args.loli_root), lol_root=Path(args.lol_root),
        max_samples=args.max_samples, mix_ratio=args.mix_ratio,
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    print(f"    LoLI : {info['n_loli']} / pool {info['loli_pool']}")
    print(f"    LOL-v2 : {info['n_lol']} / pool {info['lol_pool']}")
    print(f"    total : {info['n_total']} (batch={args.batch_size}, "
          f"steps/epoch ≈ {info['n_total']//max(args.batch_size,1)})")

    print(" [2/4] 평가 로더 (LoLI val + LOL eval15)")
    loli_val_loader, lol15_loader, n_val, n_e15 = build_eval_loaders(
        loli_root=Path(args.loli_root),
        eval_lol_root=Path(args.eval_lol_root),
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_subset_size=args.val_subset_size,
        seed=args.seed,
    )
    print(f"    LoLI val   : {n_val}")
    print(f"    LOL eval15 : {n_e15}")

    print(" [3/4] Resume baseline 측정 (LOL eval15)")
    baseline_psnr = _measure_baseline_eval15(
        Path(args.resume), lol15_loader, device=args.device,
    )
    if baseline_psnr is not None:
        print(f"    baseline LOL eval15 PSNR = {baseline_psnr:.3f}")
    else:
        print(f"    baseline 측정 안됨 — Δ(eval15) 는 0 으로 표시됩니다.")

    csv_logger = CsvLogger(csv_path)

    print(" [4/4] 학습 시작")

    # ---- Stage 1 ----
    overall_t0 = time.perf_counter()
    if args.stage2_only:
        print()
        print("[main] --stage2_only — Stage 1 건너뜀.")
        stage1_best = Path(args.resume)
        print(f"        Stage 2 시작 가중치: {stage1_best}")
    else:
        stage1_best = run_stage1(
            args, paths, train_loader, loli_val_loader, lol15_loader,
            baseline_eval15_psnr=baseline_psnr, csv_logger=csv_logger,
        )

    # ---- Stage 2 ----
    if args.skip_stage2:
        print()
        print("[main] --skip_stage2 — Stage 2 건너뜀.")
        stage2_best = paths["stage1_best"]
    else:
        if not stage1_best.is_file():
            print(f"[error] Stage 1 best 없음: {stage1_best}")
            return 1
        stage2_best = run_stage2(
            args, paths, stage1_best,
            train_loader, loli_val_loader, lol15_loader,
            baseline_eval15_psnr=baseline_psnr, csv_logger=csv_logger,
        )

    total_min = (time.perf_counter() - overall_t0) / 60.0
    print()
    print(HRULE)
    print(f"  완료 ✓  ({total_min:.1f} min)")
    print(f"  Stage 1 best : {paths['stage1_best']}")
    print(f"  Stage 2 best : {paths['stage2_best']}")
    print(f"  CSV log      : {csv_path}")
    if baseline_psnr is not None:
        print(f"  baseline LOL eval15 PSNR : {baseline_psnr:.3f}")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
