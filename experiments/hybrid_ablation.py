"""Hybrid DSConv / Standard-Conv 배치 ablation.

동기 (Motivation)
-----------------
* 기존 ``full`` (모든 블록 DSConv, base=32) → 135 K params, 1.65 G FLOPs,
  Stage 1 PSNR ≈ 17.24 dB.
* 기존 ``no_dsconv`` (모든 블록 standard Conv) → 986 K params, 12.8 G FLOPs,
  PSNR ≈ 21.59 dB. 임베디드(Jetson Orin Nano 15 W) 배포 어려움.

두 극단 사이에서 **블록별로 DSConv 와 standard Conv 를 선택적으로 배치**하여
PSNR / params / FLOPs 의 Pareto frontier 를 탐색하는 실험.

5 가지 hybrid 변형 (각 블록의 conv 종류)
----------------------------------------
+--------------+----+----+----+----+----+
| 블록         | v1 | v2 | v3 | v4 | v5 |
+==============+====+====+====+====+====+
| input_conv   | S  | D  | S  | D  | S  |
| enc1         | D  | D  | D  | D  | D  |
| enc2         | D  | D  | D  | D  | D  |
| enc3         | D  | D  | D  | D  | D  |
| bottleneck   | D  | S  | S  | D  | S  |
| dec3         | D  | D  | D  | D  | D  |
| dec2         | D  | D  | D  | S  | D  |
| dec1         | D  | D  | D  | S  | S  |
| output_conv  | (항상 1×1 plain Conv)   |
+--------------+----+----+----+----+----+

(D = DSConv 경량, S = 표준 Conv 강화)

학습 설정 (ablation.py 와 동일)
-------------------------------
* Stage 1 supervised: ``L = λ_L1·L1 + λ_VGG·VGG + λ_SSIM·SSIM``  (GAN 미사용)
* lr = 1e-3, CosineAnnealingLR(T_max=epochs, η_min=lr·0.01)
* base_filters = 32, image_size = 256

출력
----
* ``experiments/results/hybrid_<name>_log.csv``  — epoch 별 학습 로그
* ``experiments/results/hybrid_comparison.csv``  — 변형 종합 + 참조 행
* ``experiments/results/plot_hybrid_bars.png``    — PSNR / Params / FLOPs 막대

Resume: ``experiments/checkpoints/hybrid_<name>_complete.flag`` 가 있으면
해당 변형은 학습 건너뛰고 체크포인트만 로드해 평가.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# project root 를 import path 에 추가
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import get_eval_loader, get_train_loader
from models import LightEnhanceGenerator
from utils import benchmark_model_full

# ablation.py 의 supervised 학습 루프를 그대로 재사용
from experiments.ablation import train_variant  # type: ignore


HRULE = "=" * 100
SUBRULE = "-" * 100


# ===========================================================================
# 1. Hybrid variants
# ===========================================================================
@dataclass
class HybridVariant:
    """ablation.Variant 와 동일한 인터페이스 (name + lambda_*) 노출.

    추가로 ``conv_config`` 딕셔너리를 가짐. ``train_variant`` 는 v.name 과
    lambda_* 만 사용하므로 호환된다.
    """
    name: str
    description: str
    conv_config: Dict[str, str]
    base_filters: int = 32
    # ablation.py 의 SupervisedLoss 기본값과 동일
    lambda_l1: float = 1.0
    lambda_vgg: float = 0.5
    lambda_ssim: float = 1.0


def _all_dsconv() -> Dict[str, str]:
    """모든 hybrid 블록을 DSConv 로 채운 dict."""
    return {k: "dsconv" for k in LightEnhanceGenerator.HYBRID_BLOCK_NAMES}


def _make_cfg(**overrides: str) -> Dict[str, str]:
    cfg = _all_dsconv()
    cfg.update(overrides)
    return cfg


VARIANTS: List[HybridVariant] = [
    HybridVariant(
        name="hybrid_v1",
        description="input_conv 만 standard (feature 추출 강화)",
        conv_config=_make_cfg(input_conv="standard"),
    ),
    HybridVariant(
        name="hybrid_v2",
        description="bottleneck 만 standard (핵심 복원 경로 강화)",
        conv_config=_make_cfg(bottleneck="standard"),
    ),
    HybridVariant(
        name="hybrid_v3",
        description="입력 + bottleneck standard (v1 + v2 결합)",
        conv_config=_make_cfg(input_conv="standard", bottleneck="standard"),
    ),
    HybridVariant(
        name="hybrid_v4",
        description="decoder 후반부 standard (dec2, dec1)",
        conv_config=_make_cfg(dec2="standard", dec1="standard"),
    ),
    HybridVariant(
        name="hybrid_v5",
        description="입력 + bottleneck + 마지막 decoder standard (종합)",
        conv_config=_make_cfg(
            input_conv="standard", bottleneck="standard", dec1="standard",
        ),
    ),
]


# ===========================================================================
# 2. CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid DSConv ablation")
    p.add_argument("--data_root", type=str, default="../DataSet/LOLdataset")
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--base_filters", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eta_min_ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--results_dir", type=str, default="experiments/results")
    p.add_argument("--ckpt_dir", type=str, default="experiments/checkpoints")
    p.add_argument("--cpu_fps", action="store_true")
    p.add_argument("--only", type=str, default=None,
                   help="콤마로 구분된 hybrid_vN 만 실행")
    p.add_argument("--force", action="store_true",
                   help="기존 _complete.flag 무시")
    p.add_argument(
        "--reference_csv", type=str,
        default="experiments/results/ablation_summary.csv",
        help="기존 ablation_summary.csv 경로 (full / no_dsconv 참조 행).  없으면 hardcoded 사용.",
    )
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ===========================================================================
# 3. Reference rows (full / no_dsconv) — 기존 ablation 결과 재사용
# ===========================================================================
# 사용자 보고치 (이전 ablation_summary.csv 가 없을 경우 fallback)
HARDCODED_REFERENCE: Dict[str, Dict[str, float]] = {
    "full":      {"params": 135_494,  "flops": 1.65e9,  "psnr": 17.24,
                  "ssim": float("nan"), "lpips": float("nan"),
                  "fps_gpu": float("nan"), "fps_cpu": float("nan")},
    "no_dsconv": {"params": 986_000,  "flops": 12.8e9,  "psnr": 21.59,
                  "ssim": float("nan"), "lpips": float("nan"),
                  "fps_gpu": float("nan"), "fps_cpu": float("nan")},
}


def load_reference_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """ablation_summary.csv 에서 full / no_dsconv 행만 읽어 hybrid 표 형식으로 변환."""
    rows: List[Dict[str, Any]] = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["name"] in ("full", "no_dsconv"):
                    rows.append({
                        "name":          r["name"] + " (legacy 8-block)",
                        "description":   r.get("description", ""),
                        "conv_config":   "—",
                        "params":        float(r["params"]),
                        "flops":         float(r["flops"]),
                        "psnr":          float(r["psnr"]),
                        "ssim":          float(r["ssim"]),
                        "lpips":         float(r["lpips"]),
                        "fps_gpu":       float(r["fps_gpu"]),
                        "fps_cpu":       float(r.get("fps_cpu", "nan") or "nan"),
                    })
        if rows:
            return rows

    # fallback
    for name, vals in HARDCODED_REFERENCE.items():
        rows.append({
            "name":          name + " (legacy 8-block)",
            "description":   "reference (from prior ablation)",
            "conv_config":   "—",
            **vals,
        })
    return rows


# ===========================================================================
# 4. Tables + Plot
# ===========================================================================
def _fmt_count(n: float) -> str:
    if n != n:  # NaN
        return "—"
    if n >= 1e9: return f"{n/1e9:.2f}G"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def _fmt(x: float, digits: int = 2) -> str:
    if x != x:
        return "—"
    return f"{x:.{digits}f}"


def print_hybrid_table(rows: List[Dict[str, Any]]) -> None:
    print(HRULE)
    print(" Hybrid DSConv / Standard-Conv Ablation — LOL eval15 (Stage 1, 256×256)")
    print(HRULE)
    hdr = (f"  {'Variant':<28} {'Params':>9} {'FLOPs':>9} "
           f"{'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>7} {'FPS_G':>7}  Notes")
    print(hdr)
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*40}")
    for r in rows:
        marker = "  <—" if r["name"].startswith("hybrid_") else ""
        print(
            f"  {r['name']:<28} "
            f"{_fmt_count(r['params']):>9} "
            f"{_fmt_count(r['flops']):>9} "
            f"{_fmt(r['psnr'], 2):>7} "
            f"{_fmt(r['ssim'], 3):>7} "
            f"{_fmt(r['lpips'], 3):>7} "
            f"{_fmt(r['fps_gpu'], 0):>7}  "
            f"{(r.get('description') or '')[:38]}{marker}"
        )
    print(SUBRULE)


def plot_hybrid_bars(rows: List[Dict[str, Any]], out_path: Path) -> None:
    names  = [r["name"].replace(" (legacy 8-block)", "*") for r in rows]
    psnr   = [r["psnr"]   for r in rows]
    params = [r["params"] / 1e3 for r in rows]   # K
    flops  = [r["flops"]  / 1e9 for r in rows]   # G

    # 색: hybrid → 파랑, legacy → 회색, *full → 빨강 강조
    colors = []
    for r in rows:
        if r["name"].startswith("hybrid_"):
            colors.append("#3a7bd5")
        elif r["name"].startswith("full"):
            colors.append("#d04848")
        else:
            colors.append("#888888")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, vals, title, ylabel, fmt in [
        (axes[0], psnr,   "PSNR↑ (dB)",       "PSNR (dB)",  "%.2f"),
        (axes[1], params, "Params (K, ↓)",    "Params (K)", "%.0f"),
        (axes[2], flops,  "FLOPs (G, ↓)",     "FLOPs (G)",  "%.2f"),
    ]:
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, fmt % v, ha="center", va="bottom", fontsize=7)
    fig.suptitle("Hybrid DSConv ablation — accuracy / size / compute (* = legacy 8-block)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ===========================================================================
# 5. Main
# ===========================================================================
def main() -> int:
    args = parse_args()

    results_dir = Path(args.results_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    selected = set((args.only or "").split(",")) if args.only else None
    to_run = [v for v in VARIANTS if selected is None or v.name in selected]

    print(HRULE)
    print(" LightEnhanceGAN — Hybrid DSConv / Standard Conv Ablation")
    print(HRULE)
    print(f"  data_root    : {args.data_root}")
    print(f"  epochs       : {args.num_epochs} per variant")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  base_filters : {args.base_filters}")
    print(f"  lr           : {args.lr:.0e}  (cosine→×{args.eta_min_ratio})")
    print(f"  device       : {args.device}")
    print(f"  variants     : {[v.name for v in to_run]}")
    print(SUBRULE)

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

    for v in to_run:
        print()
        print(HRULE)
        print(f" Variant: {v.name}")
        print(f"   {v.description}")
        # 변형별 conv_config 한눈에 보이기
        cfg_str = ", ".join(f"{k}={v.conv_config[k][0].upper()}"
                            for k in LightEnhanceGenerator.HYBRID_BLOCK_NAMES)
        print(f"   conv_config: {cfg_str}  (D=dsconv, S=standard)")
        print(HRULE)

        ckpt_path = ckpt_dir / f"hybrid_{v.name}_best.pth"
        log_csv   = results_dir / f"hybrid_{v.name}_log.csv"
        flag_path = ckpt_dir / f"hybrid_{v.name}_complete.flag"

        skip_train = flag_path.exists() and ckpt_path.exists() and not args.force
        if skip_train:
            print(f"  → completed earlier, loading {ckpt_path}.")

        G = LightEnhanceGenerator(
            base_filters=v.base_filters,
            use_attention=True,
            conv_config=v.conv_config,
        )

        if not skip_train:
            t0 = time.perf_counter()
            tr = train_variant(
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
                  f"best PSNR={tr['best_psnr']:.2f}")
            flag_path.touch()

        # best 로드 후 종합 평가
        state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        G.load_state_dict(state["generator"])
        G = G.to(args.device)

        full = benchmark_model_full(
            G, eval_loader, device=args.device, compute_cpu_fps=args.cpu_fps,
        )
        row: Dict[str, Any] = {
            "name":         v.name,
            "description":  v.description,
            "conv_config":  ";".join(f"{k}={v.conv_config[k]}"
                                     for k in LightEnhanceGenerator.HYBRID_BLOCK_NAMES),
            **full,
        }
        summary_rows.append(row)

        print(f"  → params={int(row['params']):,}  "
              f"FLOPs={row['flops']/1e9:.3f}G  "
              f"PSNR={row['psnr']:.2f}  SSIM={row['ssim']:.4f}  "
              f"LPIPS={row['lpips']:.4f}  FPS_GPU={row['fps_gpu']:.0f}")

    # ---- Reference 행 (legacy full / no_dsconv) ----
    ref_rows = load_reference_rows(Path(args.reference_csv).resolve())
    all_rows = ref_rows[:1] + summary_rows + ref_rows[1:]  # full을 맨 위, no_dsconv 맨 아래

    # ---- Save CSV ----
    csv_path = results_dir / "hybrid_comparison.csv"
    csv_fields = ["name", "description", "conv_config",
                  "params", "macs", "flops",
                  "psnr", "ssim", "lpips", "fps_gpu", "fps_cpu"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            # macs 누락된 reference 행 보완
            if "macs" not in r:
                r["macs"] = r.get("flops", float("nan")) / 2.0
            writer.writerow(r)

    # ---- Print + Plot ----
    print()
    print_hybrid_table(all_rows)
    print()
    plot_hybrid_bars(all_rows, results_dir / "plot_hybrid_bars.png")
    print()
    print(HRULE)
    print(f" Total wall-time : {(time.perf_counter() - overall_t0)/60:.1f} min")
    print(f" CSV saved       : {csv_path}")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
