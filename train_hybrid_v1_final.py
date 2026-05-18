"""hybrid_v1 최종 모델 — 2-단계 풀 학습 + 자동 평가 단일 스크립트.

채택 구조 (Final architecture)
------------------------------
* base_filters = 32, attention 유지 (CA + SA)
* conv_config = {
      input_conv: "standard",   # 첫 conv 만 표준 Conv (저수준 feature 추출 강화)
      enc1~enc3: "dsconv",      # 인코더 전체 DSConv (경량)
      bottleneck: "dsconv",
      dec1~dec3: "dsconv",      # 디코더 전체 DSConv
  }
* 예상 규모: ≈ 205 K params, ≈ 1.89 G FLOPs (256×256 입력)

학습 (Two-stage training)
-------------------------
* Stage 1 (Supervised pre-training, 100 ep): SupervisedLoss = L1(1.0) +
  VGG(0.5) + SSIM(1.0).  lr = 1e-3, CosineAnnealingLR.  GAN 미사용.
* Stage 2 (GAN fine-tuning, 50 ep): Stage 1 best 가중치 위에서
  CombinedLoss = adv(0.01) + L1(1.0) + VGG(0.5) + SSIM(1.0).
  lr_g = lr_d = 1e-5.  PatchGAN + spectral_norm + instance noise.

Resume 정책
-----------
* ``hybrid_v1_stage1_last.pth`` 가 있으면 Stage 1 은 그 시점부터 이어서 학습.
* ``hybrid_v1_stage1_complete.flag`` 가 있으면 Stage 1 은 건너뛰고 Stage 2 로.
* Stage 2 도 동일한 구조 (``hybrid_v1_stage2_last/complete.flag``).
* ``--force_stage1`` / ``--force_stage2`` 로 flag 무시 가능.

학습 완료 후 자동 평가
----------------------
1. eval15 PSNR / SSIM / LPIPS 측정
2. Params / FLOPs / FPS (GPU + CPU) 측정
3. ``comparison.py`` 의 LITERATURE_ROWS 와 묶어 비교표 + LaTeX 출력
4. ``results/hybrid_v1_final/`` 에 시각 비교 PNG (low | enhanced | high) 저장

사용 예
-------
.. code-block:: bash

    python train_hybrid_v1_final.py \\
        --data_root "C:\\대학교\\Projects\\SmallSizePM_GAN_model\\DataSet\\LOLdataset"
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows 콘솔(cp949) 대응 — 한글 출력 안전
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_eval_loader, get_train_loader
from models import (
    CombinedLoss,
    DiscriminatorLoss,
    LightEnhanceGenerator,
    PatchGANDiscriminator,
    SupervisedLoss,
)
from utils import (
    TrainLogger,
    benchmark_model_full,
    evaluate,
    psnr_metric,
    save_comparison_grid,
    save_tensor_image,
)

# train.py 의 helper 재사용 (instance noise, FP32 D forward, seed)
from train import (
    _add_gaussian,
    _d_forward_fp32,
    instance_noise_std_for_epoch,
    set_seed,
)


# ===========================================================================
# Hybrid v1 사양 (절대 변경 금지)
# ===========================================================================
HYBRID_V1_CONV_CONFIG: Dict[str, str] = {
    "input_conv": "standard",
    "enc1":       "dsconv",
    "enc2":       "dsconv",
    "enc3":       "dsconv",
    "bottleneck": "dsconv",
    "dec3":       "dsconv",
    "dec2":       "dsconv",
    "dec1":       "dsconv",
}
HYBRID_V1_BASE_FILTERS = 32
HYBRID_V1_USE_ATTENTION = True

HRULE = "=" * 92
SUBRULE = "-" * 92


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="hybrid_v1 최종 — Stage 1 + Stage 2 자동 학습 + 평가",
    )

    # ---- 데이터 ----
    p.add_argument("--data_root", type=str, required=True,
                   help="LOL 데이터셋 루트 (our485/, eval15/ 포함)")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--full_resize", action="store_true",
                   help="random crop 대신 전체 리사이즈")

    # ---- Stage 1 ----
    p.add_argument("--stage1_epochs", type=int, default=100)
    p.add_argument("--lr_stage1", type=float, default=1e-3)

    # ---- Stage 2 ----
    p.add_argument("--stage2_epochs", type=int, default=50)
    p.add_argument("--lr_g_stage2", type=float, default=1e-5)
    p.add_argument("--lr_d_stage2", type=float, default=1e-5)
    p.add_argument("--lambda_adv", type=float, default=0.01)

    # ---- 공통 Loss 가중치 ----
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_perceptual", type=float, default=0.5)
    p.add_argument("--lambda_ssim", type=float, default=1.0)

    # ---- 옵티마이저 / 스케줄러 ----
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eta_min_ratio", type=float, default=0.01)

    # ---- GAN 안정화 (Stage 2 전용) ----
    p.add_argument("--d_update_freq", type=int, default=2)
    p.add_argument("--label_smoothing_real", type=float, default=0.9)
    p.add_argument("--instance_noise_std", type=float, default=0.1)
    p.add_argument("--instance_noise_decay_epochs", type=int, default=20)
    p.add_argument("--no_spectral_norm", action="store_true")

    # ---- I/O ----
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--log_dir",  type=str, default="./logs")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--final_results_dir", type=str,
                   default="./results/hybrid_v1_final")

    # ---- 주기 ----
    p.add_argument("--save_every",  type=int, default=10)
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=1)

    # ---- Resume / 강제 재시작 ----
    p.add_argument("--force_stage1", action="store_true",
                   help="기존 stage1 complete flag 무시하고 처음부터 재학습")
    p.add_argument("--force_stage2", action="store_true",
                   help="기존 stage2 complete flag 무시하고 처음부터 재학습")
    p.add_argument("--skip_train", action="store_true",
                   help="학습 전체 건너뛰고 best 체크포인트로 평가만 실행")

    # ---- 런타임 ----
    p.add_argument("--no_amp", action="store_true",
                   help="AMP 강제 비활성")
    p.add_argument("--no_cpu_fps", action="store_true",
                   help="평가 단계의 CPU FPS 측정 건너뛰기 (속도 우선)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ===========================================================================
# Generator factory — 항상 hybrid_v1 구조로 만들어짐
# ===========================================================================
def build_hybrid_v1_generator() -> LightEnhanceGenerator:
    """hybrid_v1 사양으로 Generator 생성. 다른 곳에서도 import 가능."""
    return LightEnhanceGenerator(
        base_filters=HYBRID_V1_BASE_FILTERS,
        use_attention=HYBRID_V1_USE_ATTENTION,
        conv_config=HYBRID_V1_CONV_CONFIG.copy(),
    )


# ===========================================================================
# Checkpoint helpers (conv_config 까지 저장 → 평가 단계에서 동일 구조 복원)
# ===========================================================================
def _save_stage1(
    path: Path,
    G: nn.Module, opt_g, sch_g, sc_g: GradScaler,
    epoch: int, best_psnr: float, args: argparse.Namespace,
) -> None:
    state = {
        "stage": 1,
        "epoch": epoch,
        "generator": G.state_dict(),
        "opt_g": opt_g.state_dict(),
        "sch_g": sch_g.state_dict(),
        "sc_g": sc_g.state_dict(),
        "best_psnr": best_psnr,
        "base_filters": HYBRID_V1_BASE_FILTERS,
        "use_attention": HYBRID_V1_USE_ATTENTION,
        "conv_config": HYBRID_V1_CONV_CONFIG.copy(),
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _save_stage2(
    path: Path,
    G: nn.Module, D: nn.Module,
    opt_g, opt_d, sch_g, sch_d,
    sc_g: GradScaler, sc_d: GradScaler,
    epoch: int, best_psnr: float, args: argparse.Namespace,
) -> None:
    state = {
        "stage": 2,
        "epoch": epoch,
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "sch_g": sch_g.state_dict(),
        "sch_d": sch_d.state_dict(),
        "sc_g": sc_g.state_dict(),
        "sc_d": sc_d.state_dict(),
        "best_psnr": best_psnr,
        "base_filters": HYBRID_V1_BASE_FILTERS,
        "use_attention": HYBRID_V1_USE_ATTENTION,
        "conv_config": HYBRID_V1_CONV_CONFIG.copy(),
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_hybrid_v1_from_ckpt(
    ckpt_path: Path, device: str,
) -> LightEnhanceGenerator:
    """체크포인트로부터 hybrid_v1 G 를 재구성하여 weight 로드."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    G = build_hybrid_v1_generator().to(device)
    sd = state["generator"] if "generator" in state else state
    G.load_state_dict(sd)
    G.eval()
    return G


# ===========================================================================
# Stage 1 — Generator pre-training (supervised)
# ===========================================================================
def run_stage1(args: argparse.Namespace, paths: Dict[str, Path]) -> Path:
    """Stage 1 supervised 학습. 반환: stage1_best 체크포인트 경로."""
    device = args.device
    set_seed(args.seed)

    # AMP 자동 비활성 (lr=1e-3 와 AMP 의 GradScaler overflow 회피)
    use_amp = (not args.no_amp) and device == "cuda"
    if use_amp and args.lr_stage1 > 2e-4:
        print(f"[stage1] WARN: lr={args.lr_stage1:.0e} + AMP 는 학습 정체 위험. "
              "AMP 자동 비활성.")
        use_amp = False

    print()
    print(HRULE)
    print(" ★ STAGE 1 — Generator Pre-training (Supervised, no GAN)")
    print(HRULE)
    print(f"  data_root        : {args.data_root}")
    print(f"  base_filters     : {HYBRID_V1_BASE_FILTERS}  attention=on")
    cfg_str = ", ".join(f"{k}={v[0].upper()}"
                        for k, v in HYBRID_V1_CONV_CONFIG.items())
    print(f"  conv_config      : {cfg_str}   (D=dsconv, S=standard)")
    print(f"  image_size       : {args.image_size}  full_resize={args.full_resize}")
    print(f"  epochs / lr      : {args.stage1_epochs} / {args.lr_stage1:.2e} "
          f"(cosine→×{args.eta_min_ratio})")
    print(f"  batch_size       : {args.batch_size}")
    print(f"  λ_L1 / λ_VGG / λ_SSIM : {args.lambda_l1} / "
          f"{args.lambda_perceptual} / {args.lambda_ssim}")
    print(f"  device / AMP     : {device} / {'on' if use_amp else 'off'}")
    print(SUBRULE)

    # ---- 모델 + 옵티마이저 ----
    G = build_hybrid_v1_generator().to(device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"  parameters       : {n_params:,}  ({n_params/1e3:.1f} K)")
    print(SUBRULE)

    opt_g = Adam(G.parameters(), lr=args.lr_stage1, betas=(args.beta1, args.beta2))
    sch_g = CosineAnnealingLR(
        opt_g, T_max=args.stage1_epochs,
        eta_min=args.lr_stage1 * args.eta_min_ratio,
    )
    sc_g = GradScaler(device="cuda", enabled=use_amp)
    loss_fn = SupervisedLoss(
        lambda_l1=args.lambda_l1,
        lambda_vgg=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim,
    ).to(device)

    # ---- 데이터 로더 ----
    train_loader = get_train_loader(
        args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=args.image_size,
        seed=args.seed, full_resize=args.full_resize,
    )
    eval_loader = get_eval_loader(
        args.data_root, batch_size=1,
        num_workers=min(args.num_workers, 2), image_size=args.image_size,
    )

    logger = TrainLogger(args.log_dir, args.results_dir,
                         run_name="hybrid_v1_stage1")

    # ---- Resume: stage1_last.pth 가 있으면 거기서부터 이어서 ----
    start_epoch = 1
    best_psnr: float = -float("inf")
    last_path = paths["stage1_last"]
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        G.load_state_dict(state["generator"])
        if "opt_g" in state: opt_g.load_state_dict(state["opt_g"])
        if "sch_g" in state: sch_g.load_state_dict(state["sch_g"])
        if "sc_g"  in state: sc_g.load_state_dict(state["sc_g"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_psnr = float(state.get("best_psnr", -float("inf")))
        print(f"[stage1] resume from {last_path.name} → epoch {start_epoch}, "
              f"best PSNR so far = {best_psnr:.2f}")
        if start_epoch > args.stage1_epochs:
            print(f"[stage1] 이미 {args.stage1_epochs} epoch 완료 — 학습 건너뜀.")
            paths["stage1_flag"].touch()
            return paths["stage1_best"]

    last_completed = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.stage1_epochs + 1):
            t0 = time.perf_counter()
            G.train()
            sums = {"g_loss": 0.0, "g_l1": 0.0, "g_vgg": 0.0,
                    "g_ssim": 0.0, "train_psnr": 0.0}
            n = 0

            pbar = tqdm(train_loader,
                        desc=f"S1 Ep {epoch:3d}/{args.stage1_epochs}",
                        ncols=110, leave=False, dynamic_ncols=False)
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
                sc_g.step(opt_g)
                sc_g.update()

                with torch.no_grad():
                    train_psnr = psnr_metric(fake.detach(), high)

                sums["g_loss"]     += float(loss.detach()) * bs
                sums["g_l1"]       += float(losses["l1"])  * bs
                sums["g_vgg"]      += float(losses["vgg"]) * bs
                sums["g_ssim"]     += float(losses["ssim"]) * bs
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

            train_avg = {k: v / max(n, 1) for k, v in sums.items()}
            train_avg["d_loss"] = 0.0  # logger 호환
            train_avg["g_adv"]  = 0.0

            # ---- eval ----
            eval_metrics: Optional[Dict[str, float]] = None
            if epoch % args.eval_every == 0:
                eval_metrics = evaluate(G, eval_loader, device=device)
                if eval_metrics["psnr"] > best_psnr:
                    best_psnr = eval_metrics["psnr"]
                    _save_stage1(paths["stage1_best"],
                                 G, opt_g, sch_g, sc_g, epoch, best_psnr, args)
                    logger.log_message(
                        f"  -> new best PSNR={best_psnr:.2f}  "
                        f"(saved {paths['stage1_best'].name})"
                    )

            logger.log_epoch(
                epoch=epoch, total_epochs=args.stage1_epochs,
                train_avg=train_avg, eval_metrics=eval_metrics,
                best_psnr=best_psnr,
                lr_g=opt_g.param_groups[0]["lr"], lr_d=0.0,
                epoch_sec=epoch_sec,
            )

            # 매 epoch 마지막에 last 저장 (resume 보장)
            _save_stage1(paths["stage1_last"],
                         G, opt_g, sch_g, sc_g, epoch, best_psnr, args)

            if epoch % args.sample_every == 0:
                grid = logger.save_samples(
                    epoch, G, eval_loader, device=device, n_samples=3,
                )
                logger.log_message(f"  samples   : {grid}")

            last_completed = epoch

    except KeyboardInterrupt:
        print("\n[stage1] Ctrl+C — last 저장 후 종료.")
        _save_stage1(paths["stage1_last"],
                     G, opt_g, sch_g, sc_g, last_completed, best_psnr, args)
        raise
    except Exception as exc:
        print(f"\n[stage1] 예외: {exc}")
        _save_stage1(paths["stage1_last"],
                     G, opt_g, sch_g, sc_g, last_completed, best_psnr, args)
        raise

    # 정상 완료 → flag
    paths["stage1_flag"].touch()
    print(f"[stage1] 완료. best PSNR = {best_psnr:.2f}  ({paths['stage1_best']})")
    return paths["stage1_best"]


# ===========================================================================
# Stage 2 — GAN fine-tuning
# ===========================================================================
def run_stage2(
    args: argparse.Namespace, paths: Dict[str, Path], stage1_best: Path,
) -> Path:
    """Stage 2 GAN fine-tuning. 반환: stage2_best 체크포인트 경로."""
    device = args.device
    set_seed(args.seed)

    use_spectral_norm = not args.no_spectral_norm
    use_amp = (not args.no_amp) and device == "cuda"
    # SN + AMP → NaN 위험 → 자동 비활성
    if use_spectral_norm and use_amp:
        print("[stage2] WARN: spectral_norm + AMP 는 NaN 위험. AMP 자동 비활성.")
        use_amp = False

    print()
    print(HRULE)
    print(" ★ STAGE 2 — GAN Fine-tuning (텍스처 미세 조정)")
    print(HRULE)
    print(f"  Stage 1 best     : {stage1_best}")
    print(f"  epochs           : {args.stage2_epochs}")
    print(f"  lr_g / lr_d      : {args.lr_g_stage2:.2e} / "
          f"{args.lr_d_stage2:.2e}")
    print(f"  λ_adv (작게)     : {args.lambda_adv}")
    print(f"  λ_L1 / λ_VGG / λ_SSIM : {args.lambda_l1} / "
          f"{args.lambda_perceptual} / {args.lambda_ssim}")
    print(f"  D 안정화         : update_freq={args.d_update_freq}, "
          f"label_smooth={args.label_smoothing_real}, "
          f"noise_σ={args.instance_noise_std} "
          f"(decay {args.instance_noise_decay_epochs}ep), "
          f"SN={'on' if use_spectral_norm else 'off'}")
    print(f"  device / AMP     : {device} / {'on' if use_amp else 'off'}")
    print(SUBRULE)

    # ---- 모델 ----
    G = build_hybrid_v1_generator().to(device)
    D = PatchGANDiscriminator(use_spectral_norm=use_spectral_norm).to(device)

    # ---- 옵티마이저 / 스케줄러 ----
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
    d_loss_fn = DiscriminatorLoss(
        real_label=args.label_smoothing_real,
    ).to(device)

    # ---- 데이터 ----
    train_loader = get_train_loader(
        args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=args.image_size,
        seed=args.seed, full_resize=args.full_resize,
    )
    eval_loader = get_eval_loader(
        args.data_root, batch_size=1,
        num_workers=min(args.num_workers, 2), image_size=args.image_size,
    )

    logger = TrainLogger(args.log_dir, args.results_dir,
                         run_name="hybrid_v1_stage2")

    # ---- Resume: stage2_last → stage1_best 우선순위 ----
    start_epoch = 1
    best_psnr: float = -float("inf")
    last_path = paths["stage2_last"]
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        G.load_state_dict(state["generator"])
        D.load_state_dict(state["discriminator"])
        if "opt_g" in state: opt_g.load_state_dict(state["opt_g"])
        if "opt_d" in state: opt_d.load_state_dict(state["opt_d"])
        if "sch_g" in state: sch_g.load_state_dict(state["sch_g"])
        if "sch_d" in state: sch_d.load_state_dict(state["sch_d"])
        if "sc_g"  in state: sc_g.load_state_dict(state["sc_g"])
        if "sc_d"  in state: sc_d.load_state_dict(state["sc_d"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_psnr = float(state.get("best_psnr", -float("inf")))
        print(f"[stage2] resume from {last_path.name} → epoch {start_epoch}, "
              f"best PSNR so far = {best_psnr:.2f}")
        if start_epoch > args.stage2_epochs:
            print(f"[stage2] 이미 {args.stage2_epochs} epoch 완료 — 학습 건너뜀.")
            paths["stage2_flag"].touch()
            return paths["stage2_best"]
    else:
        # 첫 진입 — Stage 1 best 의 G 가중치만 가져옴
        s1 = torch.load(stage1_best, map_location=device, weights_only=False)
        G.load_state_dict(s1["generator"])
        s1_best = float(s1.get("best_psnr", float("nan")))
        print(f"[stage2] Stage 1 G 가중치 로드 완료 (Stage 1 best = {s1_best:.2f})")

        # 시작점 평가 — Stage 2 가 G 를 망가뜨려도 fallback 가능
        initial = evaluate(G, eval_loader, device=device)
        best_psnr = initial["psnr"]
        print(f"[stage2] 시작 시점 eval — PSNR: {initial['psnr']:.2f}, "
              f"SSIM: {initial['ssim']:.4f}")
        _save_stage2(paths["stage2_best"],
                     G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                     0, best_psnr, args)

    last_completed = start_epoch - 1
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
            n = 0
            d_n = 0

            pbar = tqdm(train_loader,
                        desc=f"S2 Ep {epoch:3d}/{args.stage2_epochs}",
                        ncols=120, leave=False, dynamic_ncols=False)
            for step, (low, high) in enumerate(pbar):
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)
                bs = low.size(0)

                with autocast(device_type="cuda", enabled=use_amp):
                    fake = G(low)

                # ----- D step -----
                do_d_step = (step % max(args.d_update_freq, 1)) == 0
                last_d: float = float("nan")
                if do_d_step:
                    for p in D.parameters(): p.requires_grad = True
                    opt_d.zero_grad(set_to_none=True)

                    real_pair = _add_gaussian(
                        torch.cat([low, high], dim=1).float(), noise_std)
                    fake_pair = _add_gaussian(
                        torch.cat([low, fake.detach()], dim=1).float(),
                        noise_std)
                    d_real = _d_forward_fp32(D, real_pair)
                    d_fake = _d_forward_fp32(D, fake_pair)
                    d_losses = d_loss_fn(d_real, d_fake)
                    loss_d = d_losses["total"]

                    sc_d.scale(loss_d).backward()
                    sc_d.step(opt_d)
                    sc_d.update()

                    last_d = float(loss_d.detach())
                    sums["d_loss"] += last_d * bs
                    d_n += bs

                # ----- G step -----
                for p in D.parameters(): p.requires_grad = False
                opt_g.zero_grad(set_to_none=True)

                g_d_input = _add_gaussian(
                    torch.cat([low, fake], dim=1).float(), noise_std)
                d_fake_for_g = _d_forward_fp32(D, g_d_input)
                with autocast(device_type="cuda", enabled=use_amp):
                    g_losses = combined_loss(d_fake_for_g, fake, high)
                    loss_g = g_losses["total"]

                sc_g.scale(loss_g).backward()
                sc_g.step(opt_g)
                sc_g.update()

                with torch.no_grad():
                    train_psnr = psnr_metric(fake.detach(), high)

                sums["g_loss"]     += float(loss_g.detach()) * bs
                sums["g_adv"]      += float(g_losses["adv"]) * bs
                sums["g_l1"]       += float(g_losses["l1"])  * bs
                sums["g_vgg"]      += float(g_losses["vgg"]) * bs
                sums["g_ssim"]     += float(g_losses["ssim"]) * bs
                sums["train_psnr"] += train_psnr * bs
                n += bs

                pbar.set_postfix({
                    "D":     f"{last_d:.3f}" if do_d_step else "  -  ",
                    "G":     f"{float(loss_g):.3f}",
                    "L1":    f"{float(g_losses['l1']):.3f}",
                    "PSNR":  f"{train_psnr:.1f}",
                    "noise": f"{noise_std:.3f}",
                })

            pbar.close()
            sch_g.step(); sch_d.step()
            epoch_sec = time.perf_counter() - t0

            train_avg = {k: v / max(n, 1) for k, v in sums.items()
                         if k != "d_loss"}
            train_avg["d_loss"] = sums["d_loss"] / max(d_n, 1)
            train_avg["noise_std"] = noise_std

            # ---- eval ----
            eval_metrics: Optional[Dict[str, float]] = None
            if epoch % args.eval_every == 0:
                eval_metrics = evaluate(G, eval_loader, device=device)
                if eval_metrics["psnr"] > best_psnr:
                    best_psnr = eval_metrics["psnr"]
                    _save_stage2(paths["stage2_best"],
                                 G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                                 epoch, best_psnr, args)
                    logger.log_message(
                        f"  -> new best PSNR={best_psnr:.2f}  "
                        f"(saved {paths['stage2_best'].name})"
                    )

            logger.log_epoch(
                epoch=epoch, total_epochs=args.stage2_epochs,
                train_avg=train_avg, eval_metrics=eval_metrics,
                best_psnr=best_psnr,
                lr_g=opt_g.param_groups[0]["lr"],
                lr_d=opt_d.param_groups[0]["lr"],
                epoch_sec=epoch_sec,
            )

            _save_stage2(paths["stage2_last"],
                         G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                         epoch, best_psnr, args)

            if epoch % args.sample_every == 0:
                grid = logger.save_samples(
                    epoch, G, eval_loader, device=device, n_samples=3,
                )
                logger.log_message(f"  samples   : {grid}")

            last_completed = epoch

    except KeyboardInterrupt:
        print("\n[stage2] Ctrl+C — last 저장 후 종료.")
        _save_stage2(paths["stage2_last"],
                     G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                     last_completed, best_psnr, args)
        raise
    except Exception as exc:
        print(f"\n[stage2] 예외: {exc}")
        _save_stage2(paths["stage2_last"],
                     G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                     last_completed, best_psnr, args)
        raise

    paths["stage2_flag"].touch()
    print(f"[stage2] 완료. best PSNR = {best_psnr:.2f}  ({paths['stage2_best']})")
    return paths["stage2_best"]


# ===========================================================================
# Auto-evaluation
# ===========================================================================
def _fmt_count(n: Optional[float]) -> str:
    if n is None or n != n:
        return "—"
    if n >= 1e9: return f"{n/1e9:.2f}G"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def _fmt_float(x: Optional[float], digits: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:.{digits}f}"


def _print_plain_table(rows: List[Dict[str, Any]]) -> None:
    print(HRULE)
    print(" 비교표 — LOL eval15 (256×256)")
    print(HRULE)
    hdr = (f"  {'Method':<26} {'Venue':<10} {'Params':>8} {'FLOPs':>8} "
           f"{'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>7} "
           f"{'FPS_G':>7} {'FPS_C':>7}  Source")
    print(hdr)
    print(f"  {'-'*26} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*7} {'-'*7}  {'-'*9}")
    for r in rows:
        marker = "  <—" if r["source"] == "this work" else ""
        print(
            f"  {r['method']:<26} {r['venue']:<10} "
            f"{_fmt_count(r['params']):>8} {_fmt_count(r['flops']):>8} "
            f"{_fmt_float(r['psnr']):>7} {_fmt_float(r['ssim'], 3):>7} "
            f"{_fmt_float(r['lpips'], 3):>7} "
            f"{_fmt_float(r['fps_gpu'], 0):>7} "
            f"{_fmt_float(r['fps_cpu'], 1):>7}  "
            f"{r['source']}{marker}"
        )
    print(SUBRULE)


def _print_latex_table(rows: List[Dict[str, Any]]) -> None:
    print()
    print("% --- LaTeX table (논문에 그대로 복붙) ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Quantitative comparison on LOL eval15 (256$\times$256).}")
    print(r"\label{tab:hybrid_v1_comparison}")
    print(r"\small")
    print(r"\begin{tabular}{l c r r c c c r}")
    print(r"\toprule")
    print(r"Method & Venue & Params & FLOPs & PSNR$\uparrow$ & "
          r"SSIM$\uparrow$ & LPIPS$\downarrow$ & FPS \\")
    print(r"\midrule")
    for r in rows:
        bo, bc = (r"\textbf{", "}") if r["source"] == "this work" else ("", "")
        print(
            f"{bo}{r['method']}{bc} & "
            f"{r['venue']} & "
            f"{_fmt_count(r['params'])} & "
            f"{_fmt_count(r['flops'])} & "
            f"{_fmt_float(r['psnr'])} & "
            f"{_fmt_float(r['ssim'], 3)} & "
            f"{_fmt_float(r['lpips'], 3)} & "
            f"{_fmt_float(r['fps_gpu'], 0)} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("% --- end LaTeX ---")
    print()


@torch.no_grad()
def _save_final_visualizations(
    G: nn.Module, eval_loader, out_dir: Path,
    device: str, max_samples: int = 15,
) -> None:
    """eval15 전 샘플(또는 max_samples 까지)에 대해 (low | enhanced | high)
    개별 PNG + 합성 그리드 PNG 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = G.training
    G.eval()

    tensors: List[torch.Tensor] = []
    saved = 0 
    for low, high in eval_loader:
        if saved >= max_samples:
            break
        low_d = low.to(device, non_blocking=True)
        high_d = high.to(device, non_blocking=True)
        fake = G(low_d).clamp(-1.0, 1.0)

        for b in range(low_d.size(0)):
            if saved >= max_samples:
                break
            idx = saved
            save_tensor_image(low_d[b].cpu(),
                              out_dir / f"sample{idx:02d}_low.png")
            save_tensor_image(fake[b].cpu(),
                              out_dir / f"sample{idx:02d}_enhanced.png")
            save_tensor_image(high_d[b].cpu(),
                              out_dir / f"sample{idx:02d}_high.png")
            tensors.extend([low_d[b].cpu(), fake[b].cpu(), high_d[b].cpu()])
            saved += 1

    grid_path = out_dir / "comparison_grid.png"
    save_comparison_grid(tensors, grid_path, ncols=3, pad=4)
    print(f"  saved {saved} samples + grid → {out_dir}")
    if was_training:
        G.train()


def run_auto_evaluation(
    args: argparse.Namespace, paths: Dict[str, Path],
) -> None:
    """Stage 2 완료 후 자동 평가 — 지표 + 비교표 + 시각화."""
    device = args.device

    print()
    print(HRULE)
    print(" ★ FINAL EVALUATION — eval15 지표 + 비교표 + 시각화")
    print(HRULE)

    ckpt = paths["stage2_best"]
    if not ckpt.is_file():
        # Stage 2 가 한 번도 best 를 못 만든 경우 stage1_best 로 fallback
        ckpt = paths["stage1_best"]
        print(f"  WARN: Stage 2 best 가 없어 Stage 1 best 로 평가: {ckpt}")
    else:
        print(f"  checkpoint : {ckpt}")
    print(f"  data_root  : {args.data_root}")
    print(f"  device     : {device}")
    print(SUBRULE)

    G = _load_hybrid_v1_from_ckpt(ckpt, device=device)

    eval_loader = get_eval_loader(
        args.data_root, batch_size=1,
        num_workers=min(args.num_workers, 2),
        image_size=args.image_size,
    )

    # ---- 종합 벤치마크 ----
    metrics = benchmark_model_full(
        G, eval_loader, device=device,
        compute_cpu_fps=not args.no_cpu_fps,
    )

    print(f"  Params         : {int(metrics['params']):,}  "
          f"({metrics['params']/1e3:.1f} K)")
    print(f"  FLOPs (MACs×2) : {metrics['flops']/1e9:.3f} G  "
          f"(MACs={metrics['macs']/1e9:.3f} G)")
    print(f"  PSNR / SSIM    : {metrics['psnr']:.2f} dB / "
          f"{metrics['ssim']:.4f}")
    print(f"  LPIPS          : {metrics['lpips']:.4f}")
    print(f"  FPS GPU / CPU  : {metrics['fps_gpu']:.0f} / "
          f"{metrics['fps_cpu']:.1f}")
    print(SUBRULE)

    # ---- comparison.py 의 LITERATURE_ROWS 와 묶어 표 ----
    # 모듈 단위 import — comparison.py 는 main() 외에는 부작용 없음
    sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
    try:
        from experiments.comparison import LITERATURE_ROWS, LiteratureRow
    except Exception:
        # fallback: 직접 정의 (네트워크 / 패키지 문제 회피)
        from dataclasses import dataclass

        @dataclass
        class LiteratureRow:  # type: ignore[no-redef]
            method: str
            params: Optional[float] = None
            flops: Optional[float] = None
            psnr: Optional[float] = None
            ssim: Optional[float] = None
            lpips: Optional[float] = None
            fps_gpu: Optional[float] = None
            fps_cpu: Optional[float] = None
            venue: str = ""
            source: str = ""

            def as_row(self):  # type: ignore[no-untyped-def]
                return {k: getattr(self, k) for k in
                        ("method", "venue", "params", "flops", "psnr", "ssim",
                         "lpips", "fps_gpu", "fps_cpu", "source")}
        LITERATURE_ROWS = []

    rows: List[Dict[str, Any]] = [r.as_row() for r in LITERATURE_ROWS]
    rows.append(LiteratureRow(
        method="Ours (hybrid_v1)", venue="this work",
        params=metrics["params"], flops=metrics["flops"],
        psnr=metrics["psnr"], ssim=metrics["ssim"], lpips=metrics["lpips"],
        fps_gpu=metrics["fps_gpu"], fps_cpu=metrics["fps_cpu"],
        source="this work",
    ).as_row())

    _print_plain_table(rows)
    _print_latex_table(rows)

    # ---- CSV 저장 ----
    final_dir = Path(args.final_results_dir).resolve()
    final_dir.mkdir(parents=True, exist_ok=True)
    csv_path = final_dir / "comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  → {csv_path}")

    # ---- 단일 행 요약 (논문 textual report 용) ----
    summary_path = final_dir / "metrics_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("hybrid_v1 최종 모델 — eval15 지표 요약\n")
        f.write("=" * 60 + "\n")
        f.write(f"checkpoint     : {ckpt}\n")
        f.write(f"conv_config    : {HYBRID_V1_CONV_CONFIG}\n")
        f.write(f"base_filters   : {HYBRID_V1_BASE_FILTERS}\n")
        f.write(f"params         : {int(metrics['params']):,}\n")
        f.write(f"MACs           : {metrics['macs']/1e9:.4f} G\n")
        f.write(f"FLOPs (=2·MACs): {metrics['flops']/1e9:.4f} G\n")
        f.write(f"PSNR           : {metrics['psnr']:.4f} dB\n")
        f.write(f"SSIM           : {metrics['ssim']:.4f}\n")
        f.write(f"LPIPS          : {metrics['lpips']:.4f}\n")
        f.write(f"FPS (GPU)      : {metrics['fps_gpu']:.2f}\n")
        f.write(f"FPS (CPU)      : {metrics['fps_cpu']:.2f}\n")
    print(f"  TXT  → {summary_path}")

    # ---- 시각 비교 이미지 ----
    print(SUBRULE)
    print(" 시각 비교 이미지 저장 중…")
    _save_final_visualizations(
        G, eval_loader, final_dir, device=device, max_samples=15,
    )
    print(HRULE)


# ===========================================================================
# Main — Stage 1 → Stage 2 → Evaluation
# ===========================================================================
def main() -> int:
    args = parse_args()

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 표준 경로 묶음 (resume / flag / best / last)
    paths: Dict[str, Path] = {
        "stage1_best":  save_dir / "hybrid_v1_stage1_best.pth",
        "stage1_last":  save_dir / "hybrid_v1_stage1_last.pth",
        "stage1_flag":  save_dir / "hybrid_v1_stage1_complete.flag",
        "stage2_best":  save_dir / "hybrid_v1_stage2_best.pth",
        "stage2_last":  save_dir / "hybrid_v1_stage2_last.pth",
        "stage2_flag":  save_dir / "hybrid_v1_stage2_complete.flag",
    }

    print()
    print(HRULE)
    print(" hybrid_v1 최종 — 2-단계 풀 학습 자동 실행")
    print(HRULE)
    print(f"  save_dir         : {save_dir}")
    print(f"  device           : {args.device}")
    print(f"  GPU              : "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    print(SUBRULE)

    overall_t0 = time.perf_counter()

    # ---- force 옵션 처리: flag/last 삭제 후 진행 ----
    if args.force_stage1:
        for k in ("stage1_best", "stage1_last", "stage1_flag",
                  "stage2_best", "stage2_last", "stage2_flag"):
            if paths[k].exists():
                paths[k].unlink()
        print("[force_stage1] Stage 1/2 의 모든 체크포인트·flag 삭제 완료.")
    elif args.force_stage2:
        for k in ("stage2_best", "stage2_last", "stage2_flag"):
            if paths[k].exists():
                paths[k].unlink()
        print("[force_stage2] Stage 2 체크포인트·flag 삭제 완료.")

    # ---- Stage 1 ----
    if args.skip_train:
        print("[skip_train] 학습 전체 건너뛰고 평가만 실행.")
    else:
        if paths["stage1_flag"].exists() and paths["stage1_best"].is_file():
            print(f"[stage1] complete.flag 발견 — Stage 1 건너뜀. "
                  f"({paths['stage1_best']})")
        else:
            run_stage1(args, paths)

        # ---- Stage 2 ----
        if not paths["stage1_best"].is_file():
            print(f"[error] Stage 1 best 가 존재하지 않습니다: {paths['stage1_best']}")
            return 1

        if paths["stage2_flag"].exists() and paths["stage2_best"].is_file():
            print(f"[stage2] complete.flag 발견 — Stage 2 건너뜀. "
                  f"({paths['stage2_best']})")
        else:
            run_stage2(args, paths, stage1_best=paths["stage1_best"])

    # ---- 자동 평가 ----
    run_auto_evaluation(args, paths)

    total_min = (time.perf_counter() - overall_t0) / 60.0
    print()
    print(HRULE)
    print(f"  전체 소요 시간   : {total_min:.1f} min")
    print(f"  Stage 1 best     : {paths['stage1_best']}")
    print(f"  Stage 2 best     : {paths['stage2_best']}")
    print(f"  최종 결과 폴더   : {Path(args.final_results_dir).resolve()}")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
