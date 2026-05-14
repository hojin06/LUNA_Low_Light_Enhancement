"""Ablation Study 자동 실행 스크립트.

5 변형을 각각 supervised (Stage 1 방식) 50 epoch 학습 후 비교.

| name           | base | attention | dsconv | loss               |
|----------------|------|-----------|--------|--------------------|
| full           | 32   | on        | DSConv | L1 + VGG + SSIM    |
| no_attention   | 32   | off       | DSConv | L1 + VGG + SSIM    |
| no_dsconv      | 32   | on        | Conv   | L1 + VGG + SSIM    |
| small_channels | 24   | on        | DSConv | L1 + VGG + SSIM    |
| no_ssim_loss   | 32   | on        | DSConv | L1 + VGG           |

각 실험마다 ``experiments/results/`` 에 ``<name>_log.csv`` (epoch 별)
저장 후 마지막에 ``ablation_summary.csv`` (변형별 종합 지표) 출력.

Resume: ``experiments/checkpoints/ablation_<name>_complete.flag`` 가 있으면
해당 변형은 건너뛰고 체크포인트만 로드하여 평가 단계만 실행한다.

사용법
------
.. code-block:: bash

    python experiments/ablation.py \
        --data_root "C:\\대학교\\Projects\\SmallSizePM_GAN_model\\DataSet\\LOLdataset" \
        --num_epochs 50 --batch_size 8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# 부모 디렉토리(project root)를 import path 에 추가 → data/, models/, utils/ 사용
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Windows 콘솔 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_eval_loader, get_train_loader
from models import LightEnhanceGenerator, SupervisedLoss
from utils import benchmark_model_full, evaluate, psnr_metric


HRULE = "=" * 82
SUBRULE = "-" * 82


# ===========================================================================
# 1. Variants
# ===========================================================================
@dataclass
class Variant:
    name: str
    use_attention: bool = True
    use_dsconv: bool = True
    base_filters: int = 32
    lambda_l1: float = 1.0
    lambda_vgg: float = 0.5
    lambda_ssim: float = 1.0
    description: str = ""


VARIANTS: List[Variant] = [
    Variant(name="full",
            description="현재 모델 (baseline): DSConv + CA+SA + base=32, L1+VGG+SSIM"),
    Variant(name="no_attention", use_attention=False,
            description="Attention 모듈 (CA+SA) 제거"),
    Variant(name="no_dsconv", use_dsconv=False,
            description="DSConv → 표준 Conv2d 교체"),
    Variant(name="small_channels", base_filters=24,
            description="base_filters 24 (원래 경량 설계)"),
    Variant(name="no_ssim_loss", lambda_ssim=0.0,
            description="SSIM loss 제거 (L1 + VGG 만)"),
]


def build_generator(v: Variant) -> LightEnhanceGenerator:
    """Variant 사양대로 G 생성."""
    return LightEnhanceGenerator(
        base_filters=v.base_filters,
        use_attention=v.use_attention,
        use_dsconv=v.use_dsconv,
    )


# ===========================================================================
# 2. CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LightEnhanceGAN — Ablation Study")
    p.add_argument("--data_root", type=str,
                   default="../DataSet/LOLdataset")
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Stage 1 supervised 학습률")
    p.add_argument("--eta_min_ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--results_dir", type=str,
                   default="experiments/results")
    p.add_argument("--ckpt_dir", type=str,
                   default="experiments/checkpoints")
    p.add_argument("--cpu_fps", action="store_true",
                   help="CPU FPS 도 측정 (느림)")
    p.add_argument("--only", type=str, default=None,
                   help="콤마로 구분된 변형 이름만 실행 (예: full,no_attention)")
    p.add_argument("--force", action="store_true",
                   help="기존 _complete.flag 무시하고 재학습")
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ===========================================================================
# 3. Training loop (Stage 1, supervised only)
# ===========================================================================
def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_variant(
    v: Variant,
    G: nn.Module,
    train_loader,
    eval_loader,
    num_epochs: int,
    lr: float,
    eta_min_ratio: float,
    save_path: Path,
    log_csv: Path,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """변형 한 개에 대한 Stage 1 학습 + 매 epoch eval + 최고 PSNR 저장."""
    _set_seed(seed)
    G = G.to(device)

    opt = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    sch = CosineAnnealingLR(opt, T_max=num_epochs,
                            eta_min=lr * eta_min_ratio)
    loss_fn = SupervisedLoss(
        lambda_l1=v.lambda_l1,
        lambda_vgg=v.lambda_vgg,
        lambda_ssim=v.lambda_ssim,
    ).to(device)

    # CSV 헤더 준비
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "epoch", "g_loss", "g_l1", "g_vgg", "g_ssim_loss",
        "train_psnr", "eval_psnr", "eval_ssim", "lr", "epoch_sec",
    ]
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_fields)

    best_psnr = -float("inf")
    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        G.train()

        sums = {"total": 0.0, "l1": 0.0, "vgg": 0.0, "ssim": 0.0,
                "train_psnr": 0.0}
        n = 0

        pbar = tqdm(train_loader,
                    desc=f"[{v.name}] Ep {epoch:2d}/{num_epochs}",
                    ncols=110, leave=False, dynamic_ncols=False)
        for low, high in pbar:
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)
            bs = low.size(0)

            opt.zero_grad(set_to_none=True)
            fake = G(low)
            losses = loss_fn(fake, high)
            losses["total"].backward()
            opt.step()

            with torch.no_grad():
                train_psnr = psnr_metric(fake.detach(), high)

            sums["total"]      += float(losses["total"].detach()) * bs
            sums["l1"]         += float(losses["l1"]) * bs
            sums["vgg"]        += float(losses["vgg"]) * bs
            sums["ssim"]       += float(losses["ssim"]) * bs
            sums["train_psnr"] += train_psnr * bs
            n += bs

            pbar.set_postfix({
                "L":    f"{float(losses['total']):.3f}",
                "L1":   f"{float(losses['l1']):.3f}",
                "PSNR": f"{train_psnr:.1f}",
            })
        pbar.close()
        sch.step()
        epoch_sec = time.perf_counter() - t0
        avg = {k: v_ / max(n, 1) for k, v_ in sums.items()}

        # eval
        eval_m = evaluate(G, eval_loader, device=device)
        is_best = eval_m["psnr"] > best_psnr
        if is_best:
            best_psnr = eval_m["psnr"]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # 체크포인트 메타데이터: ablation.Variant 와 hybrid_ablation.HybridVariant
            # 모두를 호환적으로 받기 위해 getattr fallback 사용.
            ckpt = {
                "generator":   G.state_dict(),
                "base_filters": getattr(v, "base_filters", 32),
                "epoch":       epoch,
                "best_psnr":   best_psnr,
                "variant":     v.name,
            }
            if hasattr(v, "use_attention"):
                ckpt["use_attention"] = v.use_attention
            if hasattr(v, "use_dsconv"):
                ckpt["use_dsconv"] = v.use_dsconv
            if hasattr(v, "conv_config"):
                ckpt["conv_config"] = v.conv_config
            torch.save(ckpt, save_path)

        # CSV
        with log_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                f"{avg['total']:.6f}",
                f"{avg['l1']:.6f}",
                f"{avg['vgg']:.6f}",
                f"{avg['ssim']:.6f}",
                f"{avg['train_psnr']:.4f}",
                f"{eval_m['psnr']:.4f}",
                f"{eval_m['ssim']:.4f}",
                f"{opt.param_groups[0]['lr']:.6e}",
                f"{epoch_sec:.2f}",
            ])

        flag = " *best" if is_best else ""
        print(f"  [{v.name}] Ep {epoch:2d}/{num_epochs}  "
              f"L1={avg['l1']:.4f}  PSNR(eval)={eval_m['psnr']:.2f}  "
              f"SSIM(eval)={eval_m['ssim']:.4f}  best={best_psnr:.2f}{flag}  "
              f"({epoch_sec:.1f}s)")

    return {"best_psnr": best_psnr}


# ===========================================================================
# 4. Main
# ===========================================================================
def main() -> int:
    args = parse_args()

    results_dir = Path(args.results_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    selected = set((args.only or "").split(",")) if args.only else None
    variants_to_run = [v for v in VARIANTS
                       if selected is None or v.name in selected]

    print(HRULE)
    print(" LightEnhanceGAN — Ablation Study")
    print(HRULE)
    print(f"  data_root    : {args.data_root}")
    print(f"  epochs       : {args.num_epochs} (per variant)")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  lr           : {args.lr:.0e}  (cosine→×{args.eta_min_ratio})")
    print(f"  device       : {args.device}")
    print(f"  variants     : {[v.name for v in variants_to_run]}")
    print(f"  results_dir  : {results_dir}")
    print(f"  ckpt_dir     : {ckpt_dir}")
    print(SUBRULE)

    # 데이터 로더 한 번만 생성 (변형끼리 공유)
    train_loader = get_train_loader(
        args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=args.image_size,
        seed=args.seed,
    )
    eval_loader = get_eval_loader(
        args.data_root, batch_size=1,
        num_workers=min(args.num_workers, 2),
        image_size=args.image_size,
    )
    print(f"  train pairs  : {len(train_loader.dataset)}")
    print(f"  eval  pairs  : {len(eval_loader.dataset)}")
    print(SUBRULE)

    summary_rows: List[Dict[str, Any]] = []
    overall_t0 = time.perf_counter()

    for v in variants_to_run:
        print()
        print(HRULE)
        print(f" Variant: {v.name}")
        print(f"   {v.description}")
        print(HRULE)

        ckpt_path = ckpt_dir / f"ablation_{v.name}_best.pth"
        log_csv = results_dir / f"ablation_{v.name}_log.csv"
        flag_path = ckpt_dir / f"ablation_{v.name}_complete.flag"

        skip_train = flag_path.exists() and ckpt_path.exists() and not args.force
        if skip_train:
            print(f"  → completed, skipping training. Loading {ckpt_path}.")

        G = build_generator(v)

        if not skip_train:
            t0 = time.perf_counter()
            train_result = train_variant(
                v=v, G=G,
                train_loader=train_loader,
                eval_loader=eval_loader,
                num_epochs=args.num_epochs,
                lr=args.lr,
                eta_min_ratio=args.eta_min_ratio,
                save_path=ckpt_path,
                log_csv=log_csv,
                device=args.device,
                seed=args.seed,
            )
            elapsed = time.perf_counter() - t0
            print(f"  → variant '{v.name}' done in {elapsed/60:.1f} min, "
                  f"best PSNR={train_result['best_psnr']:.2f}")
            flag_path.touch()

        # best 체크포인트 로드 후 종합 평가
        state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        G.load_state_dict(state["generator"])
        G = G.to(args.device)

        full = benchmark_model_full(
            G, eval_loader, device=args.device,
            compute_cpu_fps=args.cpu_fps,
        )
        row: Dict[str, Any] = {
            "name":          v.name,
            "description":   v.description,
            "use_attention": v.use_attention,
            "use_dsconv":    v.use_dsconv,
            "base_filters":  v.base_filters,
            "lambda_l1":     v.lambda_l1,
            "lambda_vgg":    v.lambda_vgg,
            "lambda_ssim":   v.lambda_ssim,
            **full,
        }
        summary_rows.append(row)

        print(f"  → params={int(row['params']):,}  "
              f"FLOPs={row['flops']/1e9:.3f}G  "
              f"PSNR={row['psnr']:.2f}  SSIM={row['ssim']:.4f}  "
              f"LPIPS={row['lpips']:.4f}  FPS_GPU={row['fps_gpu']:.0f}"
              + (f"  FPS_CPU={row['fps_cpu']:.1f}" if args.cpu_fps else ""))

    # ---- Summary CSV ----
    if summary_rows:
        summary_path = results_dir / "ablation_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print()
        print(HRULE)
        print(f" Ablation summary -> {summary_path}")
        print(f" Total wall-time   : {(time.perf_counter() - overall_t0)/60:.1f} min")
        print(HRULE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
