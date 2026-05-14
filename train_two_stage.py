"""2-단계 학습 스크립트 (Two-stage training).

배경 (Why two-stage?)
---------------------
1-단계 GAN 학습은 LOL 데이터셋처럼 입력/출력의 색·조도 통계가 크게 다를 때
D 가 G 를 압도하여 PSNR 이 14~16 dB 부근에서 정체되는 경향이 강하다 (관측).
ESRGAN [Wang et al., 2018, ECCVW], pix2pixHD [Wang et al., 2018, CVPR] 등
유명 image-to-image 연구도 동일 패턴을 보고하며 ``PSNR-oriented pre-training``
이후 GAN fine-tuning 의 2-단계 전략을 권장한다.

전략
----
* **Stage 1**: Discriminator 없이 Generator 를 supervised loss
  (L1 + VGG + SSIM) 만으로 학습. lr=1e-3 으로 빠르게 강한 baseline 확보.
* **Stage 2**: Stage 1 G 가중치 위에서 GAN fine-tuning. λ_adv=0.01 (매우 작게)
  로 텍스처 디테일만 살짝 보강. 기존 GAN 안정화(label smoothing, instance noise,
  spectral norm, D update freq) 그대로 사용. lr_g, lr_d 모두 1e-5.

사용법
------
.. code-block:: bash

    # Stage 1: Generator pre-training
    python train_two_stage.py --stage 1 --data_root ../DataSet/LOLdataset \
        --num_epochs 100 --base_filters 32

    # Stage 2: GAN fine-tuning (Stage 1 best 가중치 이어서)
    python train_two_stage.py --stage 2 --data_root ../DataSet/LOLdataset \
        --num_epochs 50 --resume checkpoints/stage1_best.pth
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Windows 콘솔(cp949) 환경에서도 한글/유니코드 출력 안전.
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
from utils import TrainLogger, evaluate, psnr_metric

# train.py 의 helper 재사용
from train import (
    _add_gaussian,
    _d_forward_fp32,
    instance_noise_std_for_epoch,
    set_seed,
)


HRULE = "=" * 82
SUBRULE = "-" * 82


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LightEnhanceGAN 2-stage 학습")

    p.add_argument("--stage", type=int, choices=[1, 2], required=True,
                   help="1 = supervised pre-training, 2 = GAN fine-tuning")

    # ---- 데이터 ----
    p.add_argument("--data_root", type=str, default="../DataSet/LOLdataset")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--full_resize", action="store_true",
                   help="random crop 대신 전체 이미지 리사이즈")

    # ---- 모델 ----
    p.add_argument("--base_filters", type=int, default=32)

    # ---- 학습 ----
    p.add_argument("--num_epochs", type=int, default=None,
                   help="기본값: stage 1 = 100, stage 2 = 50")
    p.add_argument("--lr", type=float, default=None,
                   help="기본값: stage 1 = 1e-3, stage 2 = 1e-5 (G)")
    p.add_argument("--lr_d", type=float, default=1e-5,
                   help="Stage 2 D 학습률")
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eta_min_ratio", type=float, default=0.01)

    # ---- Loss weights ----
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_perceptual", type=float, default=0.5)
    p.add_argument("--lambda_ssim", type=float, default=1.0)
    p.add_argument("--lambda_adv", type=float, default=0.01,
                   help="Stage 2 의 adversarial loss 가중치 (작게 유지)")

    # ---- I/O ----
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--resume", type=str, default=None,
                   help="Stage 2: Stage 1 best checkpoint 경로 (필수)")

    # ---- 주기 ----
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=1)

    # ---- GAN 안정화 (Stage 2) ----
    p.add_argument("--d_update_freq", type=int, default=2)
    p.add_argument("--label_smoothing_real", type=float, default=0.9)
    p.add_argument("--instance_noise_std", type=float, default=0.1)
    p.add_argument("--instance_noise_decay_epochs", type=int, default=20,
                   help="Stage 2 길이 (50) 의 ~40% 까지 noise 유지")
    p.add_argument("--no_spectral_norm", action="store_true")

    # ---- 런타임 ----
    p.add_argument("--no_amp", action="store_true",
                   help="Stage 1 에서만 의미 있음 (Stage 2 는 SN 으로 자동 off)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # Stage-specific defaults
    if args.num_epochs is None:
        args.num_epochs = 100 if args.stage == 1 else 50
    if args.lr is None:
        args.lr = 1e-3 if args.stage == 1 else 1e-5
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


# ===========================================================================
# Checkpoint helpers
# ===========================================================================
def _save_stage1_ckpt(
    path: Path,
    G: nn.Module, opt_g, sch_g, sc_g: GradScaler,
    epoch: int, best_psnr: float, args: argparse.Namespace,
) -> None:
    """Stage 1 (G only) 체크포인트 저장."""
    state = {
        "stage": 1,
        "epoch": epoch,
        "generator": G.state_dict(),
        "opt_g": opt_g.state_dict(),
        "sch_g": sch_g.state_dict(),
        "sc_g": sc_g.state_dict(),
        "best_psnr": best_psnr,
        "base_filters": args.base_filters,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _save_stage2_ckpt(
    path: Path,
    G: nn.Module, D: nn.Module,
    opt_g, opt_d, sch_g, sch_d,
    sc_g: GradScaler, sc_d: GradScaler,
    epoch: int, best_psnr: float, args: argparse.Namespace,
) -> None:
    """Stage 2 (G + D + GAN state) 체크포인트 저장."""
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
        "base_filters": args.base_filters,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_stage1_generator(
    path: Path, G: nn.Module, device: str
) -> Dict[str, Any]:
    """Stage 1 체크포인트로부터 G 가중치만 로드. base_filters 일치 검증."""
    state = torch.load(path, map_location=device, weights_only=False)

    # base_filters 일치 확인 (architecture mismatch 조기 검출)
    saved_bf = state.get("base_filters")
    if saved_bf is not None:
        # G 의 첫 layer 출력 채널이 base_filters
        first_dsconv_pw = None
        for m in G.modules():
            if isinstance(m, nn.Conv2d):
                first_dsconv_pw = m
                break
        if first_dsconv_pw is not None:
            # 그냥 state_dict 크기로 비교하는 게 더 단순함
            sd = state["generator"] if "generator" in state else state
            sample_key = next(iter(sd))
            # 그냥 mismatch 시 load 실패하도록 두고, 사용자에게 메시지만 명확하게
    sd = state["generator"] if "generator" in state else state
    try:
        G.load_state_dict(sd)
    except RuntimeError as e:
        msg = (
            f"Stage 1 체크포인트 로드 실패 — Generator architecture 가 다릅니다.\n"
            f"  현재 base_filters = {_infer_base_filters(G)}, "
            f"체크포인트의 base_filters = {saved_bf}\n"
            f"  --base_filters 옵션으로 일치시키세요.\n"
            f"  원본 에러: {e}"
        )
        raise RuntimeError(msg) from e
    return state


def _infer_base_filters(G: nn.Module) -> int:
    """첫 DSConv 의 pointwise 출력 채널 = base_filters."""
    # enc1 = DSConv(in_channels=3, c1) → enc1.pointwise.out_channels == c1
    enc1 = getattr(G, "enc1", None)
    if enc1 is not None and hasattr(enc1, "pointwise"):
        return int(enc1.pointwise.out_channels)
    return -1


# ===========================================================================
# Stage 1: Generator pre-training
# ===========================================================================
def run_stage1(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device
    use_amp = (not args.no_amp) and device == "cuda"

    # Stage 1 은 lr=1e-3 같은 높은 학습률을 쓰므로 fresh network 의 초기
    # 큰 gradient 가 FP16 에서 overflow → GradScaler 가 scale 을 계속 줄이며
    # opt.step 을 스킵하여 학습이 정체되는 현상이 관찰됨.  AMP 자동 비활성.
    if use_amp and args.lr > 2e-4:
        print(f"[stage1] WARN: lr={args.lr:.0e} 와 AMP 의 조합은 GradScaler "
              "overflow 로 학습 정체를 유발합니다. AMP 자동 비활성.")
        use_amp = False

    print(HRULE)
    print(" Stage 1 — Generator Pre-training (supervised, no GAN)")
    print(HRULE)
    print(f"  data_root      : {args.data_root}")
    print(f"  base_filters   : {args.base_filters}")
    print(f"  image_size     : {args.image_size}  full_resize={args.full_resize}")
    print(f"  epochs / lr    : {args.num_epochs} / {args.lr:.2e}  (cosine→×{args.eta_min_ratio})")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  λ_L1 / λ_VGG / λ_SSIM : {args.lambda_l1} / {args.lambda_perceptual} / {args.lambda_ssim}")
    print(f"  device / AMP   : {device} / {'on' if use_amp else 'off'}")
    print(SUBRULE)

    # ---- 구성 ----
    G = LightEnhanceGenerator(base_filters=args.base_filters).to(device)
    opt_g = Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    sch_g = CosineAnnealingLR(opt_g, T_max=args.num_epochs,
                              eta_min=args.lr * args.eta_min_ratio)
    sc_g = GradScaler(device="cuda", enabled=use_amp)

    loss_fn = SupervisedLoss(
        lambda_l1=args.lambda_l1,
        lambda_vgg=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim,
    ).to(device)

    train_loader = get_train_loader(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
        full_resize=args.full_resize,
    )
    eval_loader = get_eval_loader(
        args.data_root,
        batch_size=1,
        num_workers=min(args.num_workers, 2),
        image_size=args.image_size,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(args.log_dir, args.results_dir, run_name="stage1")

    best_psnr: float = -float("inf")
    last_completed = 0

    try:
        for epoch in range(1, args.num_epochs + 1):
            t0 = time.perf_counter()
            G.train()

            sums = {"g_loss": 0.0, "g_l1": 0.0, "g_vgg": 0.0, "g_ssim": 0.0,
                    "train_psnr": 0.0}
            n = 0

            pbar = tqdm(train_loader,
                        desc=f"S1 Epoch {epoch:3d}/{args.num_epochs}",
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
            # logger 와의 호환을 위해 GAN 전용 필드 0 으로 주입
            train_avg["d_loss"] = 0.0
            train_avg["g_adv"] = 0.0

            # ---- eval ----
            eval_metrics: Optional[Dict[str, float]] = None
            if epoch % args.eval_every == 0:
                eval_metrics = evaluate(G, eval_loader, device=device)
                if eval_metrics["psnr"] > best_psnr:
                    best_psnr = eval_metrics["psnr"]
                    _save_stage1_ckpt(save_dir / "stage1_best.pth",
                                      G, opt_g, sch_g, sc_g, epoch, best_psnr, args)
                    logger.log_message(
                        f"  -> new best PSNR={best_psnr:.2f}  (saved stage1_best.pth)"
                    )

            logger.log_epoch(
                epoch=epoch, total_epochs=args.num_epochs,
                train_avg=train_avg, eval_metrics=eval_metrics,
                best_psnr=best_psnr,
                lr_g=opt_g.param_groups[0]["lr"],
                lr_d=0.0,
                epoch_sec=epoch_sec,
            )

            if epoch % args.save_every == 0:
                ckpt = save_dir / f"stage1_epoch_{epoch:03d}.pth"
                _save_stage1_ckpt(ckpt, G, opt_g, sch_g, sc_g, epoch, best_psnr, args)
                logger.log_message(f"  checkpoint: {ckpt}")

            if epoch % args.sample_every == 0:
                grid_path = logger.save_samples(
                    epoch, G, eval_loader, device=device, n_samples=3,
                )
                logger.log_message(f"  samples   : {grid_path}")

            last_completed = epoch

    except KeyboardInterrupt:
        print("\n[stage1] Ctrl+C — saving last checkpoint and exiting...")
    except Exception as exc:
        print(f"\n[stage1] Exception during training: {exc}")
        raise
    finally:
        last_path = save_dir / "stage1_last.pth"
        _save_stage1_ckpt(last_path, G, opt_g, sch_g, sc_g,
                          last_completed, best_psnr, args)
        print(f"[stage1] last checkpoint -> {last_path}  "
              f"(epoch={last_completed}, best_psnr={best_psnr:.2f})")


# ===========================================================================
# Stage 2: GAN fine-tuning
# ===========================================================================
def run_stage2(args: argparse.Namespace) -> None:
    if args.resume is None:
        # 사용자가 명시하지 않으면 표준 경로를 자동 사용
        default_resume = Path(args.save_dir) / "stage1_best.pth"
        if default_resume.is_file():
            args.resume = str(default_resume)
            print(f"[stage2] --resume 미지정 → 자동 사용: {args.resume}")
        else:
            raise ValueError(
                "Stage 2 는 Stage 1 체크포인트가 필요합니다. "
                "--resume <path> 로 지정하거나 ./checkpoints/stage1_best.pth 를 준비하세요."
            )

    set_seed(args.seed)
    device = args.device
    use_spectral_norm = not args.no_spectral_norm
    use_amp = (not args.no_amp) and device == "cuda"

    # SN + AMP 호환성 회피 (train.py 와 동일 정책)
    if use_spectral_norm and use_amp:
        print("[stage2] WARN: spectral_norm + AMP → NaN 위험. AMP 자동 비활성.")
        use_amp = False

    print(HRULE)
    print(" Stage 2 — GAN Fine-tuning (텍스처 미세 조정)")
    print(HRULE)
    print(f"  resume (Stage 1) : {args.resume}")
    print(f"  base_filters     : {args.base_filters}")
    print(f"  epochs           : {args.num_epochs}")
    print(f"  lr_g / lr_d      : {args.lr:.2e} / {args.lr_d:.2e}")
    print(f"  λ_adv (작게)     : {args.lambda_adv}")
    print(f"  λ_L1 / λ_VGG / λ_SSIM : {args.lambda_l1} / {args.lambda_perceptual} / {args.lambda_ssim}")
    print(f"  D 안정화         : update_freq={args.d_update_freq}, "
          f"label_smooth={args.label_smoothing_real}, "
          f"noise_σ={args.instance_noise_std} "
          f"(decay {args.instance_noise_decay_epochs}ep), "
          f"SN={'on' if use_spectral_norm else 'off'}")
    print(f"  device / AMP     : {device} / {'on' if use_amp else 'off'}")
    print(SUBRULE)

    # ---- 모델 ----
    G = LightEnhanceGenerator(base_filters=args.base_filters).to(device)
    D = PatchGANDiscriminator(use_spectral_norm=use_spectral_norm).to(device)

    # Stage 1 가중치 로드 (G 만)
    state = _load_stage1_generator(Path(args.resume), G, device=device)
    print(f"[stage2] Stage 1 G weights loaded.  "
          f"(Stage 1 best PSNR = {state.get('best_psnr', float('nan')):.2f})")

    opt_g = Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    sch_g = CosineAnnealingLR(opt_g, T_max=args.num_epochs,
                              eta_min=args.lr * args.eta_min_ratio)
    sch_d = CosineAnnealingLR(opt_d, T_max=args.num_epochs,
                              eta_min=args.lr_d * args.eta_min_ratio)
    sc_g = GradScaler(device="cuda", enabled=use_amp)
    sc_d = GradScaler(device="cuda", enabled=use_amp)

    combined_loss = CombinedLoss(
        lambda_adv=args.lambda_adv,
        lambda_l1=args.lambda_l1,
        lambda_vgg=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim,
    ).to(device)
    d_loss_fn = DiscriminatorLoss(real_label=args.label_smoothing_real).to(device)

    train_loader = get_train_loader(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
        full_resize=args.full_resize,
    )
    eval_loader = get_eval_loader(
        args.data_root,
        batch_size=1,
        num_workers=min(args.num_workers, 2),
        image_size=args.image_size,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(args.log_dir, args.results_dir, run_name="stage2")

    # eval baseline (Stage 1 G 시작점)
    initial_eval = evaluate(G, eval_loader, device=device)
    print(f"[stage2] 시작 시점 eval — PSNR: {initial_eval['psnr']:.2f}, "
          f"SSIM: {initial_eval['ssim']:.4f}")
    best_psnr: float = initial_eval["psnr"]
    # 시작점 체크포인트 (Stage 2 가 G 를 망가뜨려도 복원 가능)
    _save_stage2_ckpt(save_dir / "stage2_best.pth",
                      G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                      0, best_psnr, args)

    last_completed = 0
    try:
        for epoch in range(1, args.num_epochs + 1):
            t0 = time.perf_counter()
            noise_std = instance_noise_std_for_epoch(
                epoch, args.instance_noise_std, args.instance_noise_decay_epochs,
            )

            G.train(); D.train()
            sums = {"d_loss": 0.0, "g_loss": 0.0,
                    "g_adv": 0.0, "g_l1": 0.0, "g_vgg": 0.0, "g_ssim": 0.0,
                    "train_psnr": 0.0}
            n = 0
            d_n = 0

            pbar = tqdm(train_loader,
                        desc=f"S2 Epoch {epoch:3d}/{args.num_epochs}",
                        ncols=120, leave=False, dynamic_ncols=False)
            for step, (low, high) in enumerate(pbar):
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)
                bs = low.size(0)

                with autocast(device_type="cuda", enabled=use_amp):
                    fake = G(low)

                # ----- D step (매 d_update_freq 번 마다) -----
                do_d_step = (step % max(args.d_update_freq, 1)) == 0
                last_d: float = float("nan")
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
            sch_g.step()
            sch_d.step()
            epoch_sec = time.perf_counter() - t0

            train_avg = {k: v / max(n, 1) for k, v in sums.items() if k != "d_loss"}
            train_avg["d_loss"] = sums["d_loss"] / max(d_n, 1)
            train_avg["noise_std"] = noise_std

            # ---- eval ----
            eval_metrics: Optional[Dict[str, float]] = None
            if epoch % args.eval_every == 0:
                eval_metrics = evaluate(G, eval_loader, device=device)
                if eval_metrics["psnr"] > best_psnr:
                    best_psnr = eval_metrics["psnr"]
                    _save_stage2_ckpt(save_dir / "stage2_best.pth",
                                      G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                                      epoch, best_psnr, args)
                    logger.log_message(
                        f"  -> new best PSNR={best_psnr:.2f}  (saved stage2_best.pth)"
                    )

            logger.log_epoch(
                epoch=epoch, total_epochs=args.num_epochs,
                train_avg=train_avg, eval_metrics=eval_metrics,
                best_psnr=best_psnr,
                lr_g=opt_g.param_groups[0]["lr"],
                lr_d=opt_d.param_groups[0]["lr"],
                epoch_sec=epoch_sec,
            )

            if epoch % args.save_every == 0:
                ckpt = save_dir / f"stage2_epoch_{epoch:03d}.pth"
                _save_stage2_ckpt(ckpt, G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                                  epoch, best_psnr, args)
                logger.log_message(f"  checkpoint: {ckpt}")

            if epoch % args.sample_every == 0:
                grid_path = logger.save_samples(
                    epoch, G, eval_loader, device=device, n_samples=3,
                )
                logger.log_message(f"  samples   : {grid_path}")

            last_completed = epoch

    except KeyboardInterrupt:
        print("\n[stage2] Ctrl+C — saving last checkpoint and exiting...")
    except Exception as exc:
        print(f"\n[stage2] Exception during training: {exc}")
        raise
    finally:
        last_path = save_dir / "stage2_last.pth"
        _save_stage2_ckpt(last_path, G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                          last_completed, best_psnr, args)
        print(f"[stage2] last checkpoint -> {last_path}  "
              f"(epoch={last_completed}, best_psnr={best_psnr:.2f})")


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> int:
    args = parse_args()
    if args.stage == 1:
        run_stage1(args)
    else:
        run_stage2(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
