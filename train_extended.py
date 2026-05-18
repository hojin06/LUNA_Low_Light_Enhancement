"""hybrid_v1 모델의 다중 데이터셋 통합 학습 스크립트.

기존 [train_hybrid_v1_final.py] 와 동일한 2-단계 전략(Stage 1 supervised + Stage 2
GAN fine-tuning) 을 그대로 사용하되, ``--dataset`` 인자로 학습 데이터를 선택.

지원 데이터셋
-------------
* ``lol_v1``      — LOL v1 (our485 → 학습, eval15 → 평가)
* ``lol_v2_real`` — LOL-v2 Real_captured (Train/Test)
* ``lol_v2_syn``  — LOL-v2 Synthetic (Train/Test)
* ``loli_street`` — LoLI-Street (train/test)
* ``all``         — 위 모든 데이터셋을 ``CombinedDataset`` 으로 합쳐서 학습.
                   평가는 LOL eval15 단일 (논문 비교 일관성).

Resume 옵션
-----------
* ``--resume <path>`` 가 주어지면 해당 체크포인트의 ``generator`` 가중치를
  로드하여 그 위에서 Stage 1 부터 학습.  (typical 사용: 기존 LOL v1 학습 best
  → ``checkpoints/hybrid_v1_stage2_best.pth`` → 새 데이터셋으로 fine-tune)
* 매 epoch 끝에 ``{run_tag}_stage{1,2}_last.pth`` 가 자동 저장되므로 중단 시
  같은 명령으로 이어서 재개 가능 (resume 자동).

사용 예
-------
.. code-block:: bash

    # LOL-v2 Real 로 추가 학습 (기존 hybrid_v1 best 에서 이어서)
    python train_extended.py --dataset lol_v2_real \\
        --data_root "C:\\대학교\\Projects\\SmallSizePM_GAN_model\\DataSet\\LOL-v2" \\
        --resume checkpoints/hybrid_v1_stage2_best.pth

    # 전체 데이터셋 합쳐서 학습
    python train_extended.py --dataset all \\
        --data_root "C:\\대학교\\Projects\\SmallSizePM_GAN_model\\DataSet" \\
        --resume checkpoints/hybrid_v1_stage2_best.pth
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows 콘솔 UTF-8 안전
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    LOLDataset,
    PairedAugment,
    build_combined_dataset,
    build_dataset_by_name,
)
from models import (
    CombinedLoss,
    DiscriminatorLoss,
    LightEnhanceGenerator,
    PatchGANDiscriminator,
    SupervisedLoss,
)
from utils import TrainLogger, evaluate, psnr_metric

# 학습 헬퍼는 train.py / train_hybrid_v1_final.py 의 모듈 함수 재사용
from train import (
    _add_gaussian,
    _d_forward_fp32,
    instance_noise_std_for_epoch,
    set_seed,
)
from train_hybrid_v1_final import (
    HYBRID_V1_BASE_FILTERS,
    HYBRID_V1_CONV_CONFIG,
    HYBRID_V1_USE_ATTENTION,
    build_hybrid_v1_generator,
)


HRULE = "=" * 92
SUBRULE = "-" * 92


# ===========================================================================
# 1. CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="hybrid_v1 다중 데이터셋 통합 학습 (Stage1 + Stage2)",
    )

    # ---- 데이터셋 ----
    p.add_argument("--dataset", type=str, required=True,
                   choices=["lol_v1", "lol_v2_real", "lol_v2_syn",
                            "loli_street", "all"],
                   help="학습에 사용할 데이터셋 선택.  'all' 은 모두 합침.")
    p.add_argument("--data_root", type=str, required=True,
                   help=("선택 데이터셋의 루트 폴더.  --dataset all 인 경우 "
                         "DataSet/ 의 부모 폴더 (LOLdataset, LOL-v2, "
                         "LoLI-Street 등이 자식으로 있는 폴더)."))
    p.add_argument("--eval_data_root", type=str, default=None,
                   help="평가용 LOL v1 폴더 (eval15 가 있는 폴더). 미지정 시 "
                        "--data_root + LOLdataset 자동 탐색 + fallback.")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--full_resize", action="store_true")

    # ---- 학습 ----
    p.add_argument("--stage1_epochs", type=int, default=100)
    p.add_argument("--lr_stage1", type=float, default=1e-3)
    p.add_argument("--stage2_epochs", type=int, default=50)
    p.add_argument("--lr_g_stage2", type=float, default=1e-5)
    p.add_argument("--lr_d_stage2", type=float, default=1e-5)
    p.add_argument("--lambda_adv", type=float, default=0.01)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_perceptual", type=float, default=0.5)
    p.add_argument("--lambda_ssim", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eta_min_ratio", type=float, default=0.01)

    # ---- GAN 안정화 ----
    p.add_argument("--d_update_freq", type=int, default=2)
    p.add_argument("--label_smoothing_real", type=float, default=0.9)
    p.add_argument("--instance_noise_std", type=float, default=0.1)
    p.add_argument("--instance_noise_decay_epochs", type=int, default=20)
    p.add_argument("--no_spectral_norm", action="store_true")

    # ---- I/O / 주기 ----
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--log_dir",  type=str, default="./logs")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--save_every",  type=int, default=10)
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=1)

    # ---- Resume ----
    p.add_argument("--resume", type=str, default=None,
                   help="기존 G 가중치(.pth)에서 이어서 시작. typical 사용: "
                        "checkpoints/hybrid_v1_stage2_best.pth")
    p.add_argument("--force_stage1", action="store_true")
    p.add_argument("--force_stage2", action="store_true")
    p.add_argument("--skip_train", action="store_true")

    # ---- 런타임 ----
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ===========================================================================
# 2. 데이터 로더 빌더 — 데이터셋 키 → DataLoader
# ===========================================================================
def _make_train_loader(args: argparse.Namespace) -> DataLoader:
    """``--dataset`` 인자에 따라 학습 로더 생성."""
    if args.dataset == "all":
        ds = build_combined_dataset(
            dataset_root=args.data_root, split="train",
            image_size=args.image_size, augment=True,
            full_resize=args.full_resize,
            include=("lol_v1", "lol_v2_real", "lol_v2_syn", "loli_street"),
            skip_missing=True,
        )
    else:
        ds = build_dataset_by_name(
            name=args.dataset, data_root=args.data_root, split="train",
            image_size=args.image_size, augment=True,
            full_resize=args.full_resize,
        )

    gen = torch.Generator()
    gen.manual_seed(args.seed)

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True, drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        generator=gen,
    )


def _make_eval_loader(args: argparse.Namespace) -> DataLoader:
    """평가용 로더는 일관성을 위해 LOL v1 eval15 만 사용.

    학습 데이터셋이 무엇이든 논문/표 비교의 일관성을 위해 동일한 평가셋
    (LOL eval15) 으로 PSNR/SSIM 을 측정.
    """
    # 1) 사용자 명시 경로 우선
    candidates: List[Path] = []
    if args.eval_data_root:
        candidates.append(Path(args.eval_data_root))
    # 2) --data_root 자체가 LOLdataset 일 가능성
    candidates.append(Path(args.data_root))
    # 3) --data_root 의 부모 또는 자식에 LOLdataset 가 있을 가능성
    candidates.append(Path(args.data_root) / "LOLdataset")
    candidates.append(Path(args.data_root).parent / "LOLdataset")

    eval_root: Optional[Path] = None
    for c in candidates:
        if (c / "eval15" / "low").is_dir():
            eval_root = c
            break

    if eval_root is None:
        raise FileNotFoundError(
            f"평가용 LOL eval15 를 찾을 수 없습니다. --eval_data_root <LOLdataset> "
            f"로 명시 지정하세요.\n  시도한 경로: {candidates}"
        )

    eval_ds = LOLDataset(
        data_root=eval_root, split="eval",
        image_size=args.image_size, augment=False,
    )
    return DataLoader(
        eval_ds, batch_size=1,
        num_workers=min(args.num_workers, 2), shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=min(args.num_workers, 2) > 0,
    )


# ===========================================================================
# 3. 체크포인트 helpers
# ===========================================================================
def _save_stage1(
    path: Path,
    G: nn.Module, opt_g, sch_g, sc_g: GradScaler,
    epoch: int, best_psnr: float, args: argparse.Namespace,
) -> None:
    state = {
        "stage": 1, "epoch": epoch,
        "generator": G.state_dict(),
        "opt_g": opt_g.state_dict(),
        "sch_g": sch_g.state_dict(),
        "sc_g": sc_g.state_dict(),
        "best_psnr": best_psnr,
        "base_filters": HYBRID_V1_BASE_FILTERS,
        "use_attention": HYBRID_V1_USE_ATTENTION,
        "conv_config": HYBRID_V1_CONV_CONFIG.copy(),
        "dataset": args.dataset,
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
        "stage": 2, "epoch": epoch,
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
        "dataset": args.dataset,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_g_weights(G: nn.Module, ckpt_path: Path, device: str) -> Dict[str, Any]:
    """``--resume`` 으로 지정된 체크포인트에서 G 가중치만 로드.

    config (conv_config, base_filters) 가 다른 체크포인트는 ``load_state_dict``
    가 실패하므로 명확한 에러 메시지로 보고.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["generator"] if "generator" in state else state
    try:
        G.load_state_dict(sd)
    except RuntimeError as e:
        raise RuntimeError(
            f"--resume {ckpt_path} 의 G 가중치 로드 실패 — architecture 불일치.\n"
            f"  본 스크립트는 hybrid_v1 (base_filters=32, conv_config=...) 고정.\n"
            f"  체크포인트의 base_filters = {state.get('base_filters')}\n"
            f"  체크포인트의 conv_config  = {state.get('conv_config')}\n"
            f"  원본 에러: {e}"
        ) from e
    return state


# ===========================================================================
# 4. Stage 1 — Supervised pre-training (with optional resume)
# ===========================================================================
def run_stage1(args: argparse.Namespace, paths: Dict[str, Path],
               resume_g_ckpt: Optional[Path]) -> Path:
    device = args.device
    set_seed(args.seed)

    use_amp = (not args.no_amp) and device == "cuda"
    if use_amp and args.lr_stage1 > 2e-4:
        print(f"[stage1] WARN: lr={args.lr_stage1:.0e} + AMP → overflow 위험. "
              "AMP 자동 비활성.")
        use_amp = False

    print()
    print(HRULE)
    print(f" ★ STAGE 1 — Supervised Pre-training  (dataset = {args.dataset})")
    print(HRULE)
    print(f"  data_root      : {args.data_root}")
    print(f"  epochs / lr    : {args.stage1_epochs} / {args.lr_stage1:.2e}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  resume         : {resume_g_ckpt}")
    print(f"  device / AMP   : {device} / {'on' if use_amp else 'off'}")
    print(SUBRULE)

    G = build_hybrid_v1_generator().to(device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"  parameters     : {n_params:,}")

    # ---- Resume from external G weights (if provided AND no stage1_last) ----
    if resume_g_ckpt is not None and not paths["stage1_last"].is_file():
        s = _load_g_weights(G, resume_g_ckpt, device=device)
        print(f"[stage1] resume G from {resume_g_ckpt.name}  "
              f"(prev best PSNR = {s.get('best_psnr', float('nan')):.2f})")

    opt_g = Adam(G.parameters(), lr=args.lr_stage1,
                 betas=(args.beta1, args.beta2))
    sch_g = CosineAnnealingLR(opt_g, T_max=args.stage1_epochs,
                              eta_min=args.lr_stage1 * args.eta_min_ratio)
    sc_g = GradScaler(device="cuda", enabled=use_amp)
    loss_fn = SupervisedLoss(
        lambda_l1=args.lambda_l1, lambda_vgg=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim,
    ).to(device)

    train_loader = _make_train_loader(args)
    eval_loader  = _make_eval_loader(args)
    print(f"  train pairs    : {len(train_loader.dataset)}")  # type: ignore[arg-type]
    print(f"  eval15 pairs   : {len(eval_loader.dataset)}")   # type: ignore[arg-type]

    logger = TrainLogger(args.log_dir, args.results_dir,
                         run_name=f"{paths['run_tag']}_stage1")

    # ---- Resume from stage1_last (epoch-by-epoch) ----
    start_epoch = 1
    best_psnr: float = -float("inf")
    if paths["stage1_last"].is_file():
        s = torch.load(paths["stage1_last"], map_location=device,
                       weights_only=False)
        G.load_state_dict(s["generator"])
        opt_g.load_state_dict(s["opt_g"])
        sch_g.load_state_dict(s["sch_g"])
        sc_g.load_state_dict(s["sc_g"])
        start_epoch = int(s.get("epoch", 0)) + 1
        best_psnr = float(s.get("best_psnr", -float("inf")))
        print(f"[stage1] resume from {paths['stage1_last'].name} → "
              f"epoch {start_epoch}, best PSNR={best_psnr:.2f}")
        if start_epoch > args.stage1_epochs:
            print(f"[stage1] 이미 완료 — 학습 건너뜀.")
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
                sums["g_l1"]       += float(losses["l1"]) * bs
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
            train_avg["d_loss"] = 0.0
            train_avg["g_adv"] = 0.0

            eval_metrics = None
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

            _save_stage1(paths["stage1_last"],
                         G, opt_g, sch_g, sc_g, epoch, best_psnr, args)

            if epoch % args.sample_every == 0:
                grid = logger.save_samples(epoch, G, eval_loader,
                                           device=device, n_samples=3)
                logger.log_message(f"  samples   : {grid}")

            last_completed = epoch

    except KeyboardInterrupt:
        print("\n[stage1] Ctrl+C — last 저장 후 종료.")
        _save_stage1(paths["stage1_last"],
                     G, opt_g, sch_g, sc_g, last_completed, best_psnr, args)
        raise

    paths["stage1_flag"].touch()
    print(f"[stage1] 완료. best PSNR = {best_psnr:.2f}")
    return paths["stage1_best"]


# ===========================================================================
# 5. Stage 2 — GAN fine-tuning
# ===========================================================================
def run_stage2(args: argparse.Namespace, paths: Dict[str, Path],
               stage1_best: Path) -> Path:
    device = args.device
    set_seed(args.seed)

    use_spectral_norm = not args.no_spectral_norm
    use_amp = (not args.no_amp) and device == "cuda"
    if use_spectral_norm and use_amp:
        print("[stage2] WARN: SN + AMP → NaN. AMP 자동 비활성.")
        use_amp = False

    print()
    print(HRULE)
    print(f" ★ STAGE 2 — GAN Fine-tuning  (dataset = {args.dataset})")
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
        lambda_l1=args.lambda_l1, lambda_vgg=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim,
    ).to(device)
    d_loss_fn = DiscriminatorLoss(
        real_label=args.label_smoothing_real,
    ).to(device)

    train_loader = _make_train_loader(args)
    eval_loader  = _make_eval_loader(args)

    logger = TrainLogger(args.log_dir, args.results_dir,
                         run_name=f"{paths['run_tag']}_stage2")

    # Resume: stage2_last → stage1_best 우선순위
    start_epoch = 1
    best_psnr: float = -float("inf")
    if paths["stage2_last"].is_file():
        s = torch.load(paths["stage2_last"], map_location=device,
                       weights_only=False)
        G.load_state_dict(s["generator"])
        D.load_state_dict(s["discriminator"])
        opt_g.load_state_dict(s["opt_g"]); opt_d.load_state_dict(s["opt_d"])
        sch_g.load_state_dict(s["sch_g"]); sch_d.load_state_dict(s["sch_d"])
        sc_g.load_state_dict(s["sc_g"]); sc_d.load_state_dict(s["sc_d"])
        start_epoch = int(s.get("epoch", 0)) + 1
        best_psnr = float(s.get("best_psnr", -float("inf")))
        print(f"[stage2] resume from {paths['stage2_last'].name} → "
              f"epoch {start_epoch}, best={best_psnr:.2f}")
        if start_epoch > args.stage2_epochs:
            print(f"[stage2] 이미 완료 — 학습 건너뜀.")
            paths["stage2_flag"].touch()
            return paths["stage2_best"]
    else:
        s1 = torch.load(stage1_best, map_location=device, weights_only=False)
        G.load_state_dict(s1["generator"])
        initial = evaluate(G, eval_loader, device=device)
        best_psnr = initial["psnr"]
        print(f"[stage2] Stage 1 G 로드 완료. 시작 PSNR = {best_psnr:.2f}")
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
            n = 0; d_n = 0

            pbar = tqdm(train_loader,
                        desc=f"S2 Ep {epoch:3d}/{args.stage2_epochs}",
                        ncols=120, leave=False, dynamic_ncols=False)
            for step, (low, high) in enumerate(pbar):
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)
                bs = low.size(0)

                with autocast(device_type="cuda", enabled=use_amp):
                    fake = G(low)

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
                    sc_d.step(opt_d); sc_d.update()
                    last_d = float(loss_d.detach())
                    sums["d_loss"] += last_d * bs
                    d_n += bs

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
                    "D": f"{last_d:.3f}" if do_d_step else "  -  ",
                    "G": f"{float(loss_g):.3f}",
                    "PSNR": f"{train_psnr:.1f}",
                    "noise": f"{noise_std:.3f}",
                })
            pbar.close()
            sch_g.step(); sch_d.step()
            epoch_sec = time.perf_counter() - t0

            train_avg = {k: v / max(n, 1) for k, v in sums.items()
                         if k != "d_loss"}
            train_avg["d_loss"] = sums["d_loss"] / max(d_n, 1)
            train_avg["noise_std"] = noise_std

            eval_metrics = None
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
                grid = logger.save_samples(epoch, G, eval_loader,
                                           device=device, n_samples=3)
                logger.log_message(f"  samples   : {grid}")
            last_completed = epoch

    except KeyboardInterrupt:
        print("\n[stage2] Ctrl+C — last 저장 후 종료.")
        _save_stage2(paths["stage2_last"],
                     G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                     last_completed, best_psnr, args)
        raise

    paths["stage2_flag"].touch()
    print(f"[stage2] 완료. best PSNR = {best_psnr:.2f}")
    return paths["stage2_best"]


# ===========================================================================
# 6. Main
# ===========================================================================
def main() -> int:
    args = parse_args()

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # run_tag = ext_{dataset}  →  ext_lol_v2_real / ext_all / ...
    run_tag = f"ext_{args.dataset}"

    paths: Dict[str, Path] = {
        "run_tag":      Path(run_tag),  # str 호환을 위해 Path 로 두지만 실제로는 문자열로 사용
        "stage1_best":  save_dir / f"{run_tag}_stage1_best.pth",
        "stage1_last":  save_dir / f"{run_tag}_stage1_last.pth",
        "stage1_flag":  save_dir / f"{run_tag}_stage1_complete.flag",
        "stage2_best":  save_dir / f"{run_tag}_stage2_best.pth",
        "stage2_last":  save_dir / f"{run_tag}_stage2_last.pth",
        "stage2_flag":  save_dir / f"{run_tag}_stage2_complete.flag",
    }
    paths["run_tag"] = run_tag  # type: ignore[assignment]   # 사실은 문자열

    print()
    print(HRULE)
    print(f" hybrid_v1 확장 학습  —  dataset = {args.dataset}")
    print(HRULE)
    print(f"  save_dir     : {save_dir}")
    print(f"  run_tag      : {run_tag}")
    print(f"  device       : {args.device}")
    print(f"  GPU          : "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # force 옵션 처리
    if args.force_stage1:
        for k in ("stage1_best", "stage1_last", "stage1_flag",
                  "stage2_best", "stage2_last", "stage2_flag"):
            if paths[k].exists(): paths[k].unlink()
        print("[force_stage1] 모든 체크포인트/flag 삭제 후 처음부터.")
    elif args.force_stage2:
        for k in ("stage2_best", "stage2_last", "stage2_flag"):
            if paths[k].exists(): paths[k].unlink()
        print("[force_stage2] Stage 2 체크포인트/flag 삭제 후 재시작.")

    resume_g_ckpt: Optional[Path] = None
    if args.resume:
        rp = Path(args.resume)
        if rp.is_file():
            resume_g_ckpt = rp
        else:
            print(f"[warn] --resume 파일을 찾을 수 없음: {rp} (무시)")

    overall_t0 = time.perf_counter()

    # Stage 1 → 2
    if not args.skip_train:
        if paths["stage1_flag"].exists() and paths["stage1_best"].is_file():
            print(f"[stage1] complete.flag 발견 — 건너뜀.")
        else:
            run_stage1(args, paths, resume_g_ckpt=resume_g_ckpt)

        if not paths["stage1_best"].is_file():
            print(f"[error] Stage 1 best 없음: {paths['stage1_best']}")
            return 1

        if paths["stage2_flag"].exists() and paths["stage2_best"].is_file():
            print(f"[stage2] complete.flag 발견 — 건너뜀.")
        else:
            run_stage2(args, paths, stage1_best=paths["stage1_best"])

    total_min = (time.perf_counter() - overall_t0) / 60.0
    print()
    print(HRULE)
    print(f"  전체 시간      : {total_min:.1f} min")
    print(f"  Stage 1 best   : {paths['stage1_best']}")
    print(f"  Stage 2 best   : {paths['stage2_best']}")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
