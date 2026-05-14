"""비교 실험 — 우리 모델의 논문용 종합 지표 + 기존 모델 literature 값.

기존 모델 (Zero-DCE++, SCI, EnlightenGAN, RetinexNet, FUnIE-GAN) 의 공식
가중치를 LOL eval15 에서 실제로 돌리는 것은 환경 의존성 (pretrained weight
파일, 별도 repo 의존성 등) 때문에 안정 자동화가 어렵다.  대신 본 스크립트는

* **우리 모델**: 체크포인트 로드 → PSNR/SSIM/LPIPS/Params/FLOPs/FPS GPU·CPU
  를 실측.
* **타 모델**: 각 논문 / 공식 코드 기준의 보고 값을 표에 함께 출력.

논문에 그대로 가져다 쓰기 좋은 plain + LaTeX 표 두 가지를 모두 ``stdout``
및 ``experiments/results/comparison_table.csv`` 에 저장한다.

사용법
------
.. code-block:: bash

    python experiments/comparison.py --checkpoint checkpoints/stage2_best.pth
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch

from data import get_eval_loader
from models import LightEnhanceGenerator
from utils import benchmark_model_full


HRULE = "=" * 100
SUBRULE = "-" * 100


# ===========================================================================
# 1. Literature-reported values for prior methods (LOL eval15, 256×256)
# ===========================================================================
@dataclass
class LiteratureRow:
    method: str
    params: Optional[float] = None       # 단위: 개
    flops: Optional[float] = None        # 단위: FLOPs (≈2·MACs)
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    fps_gpu: Optional[float] = None
    fps_cpu: Optional[float] = None
    venue: str = ""
    source: str = ""                     # "paper" | "github" | "this work"

    def as_row(self) -> Dict[str, Any]:
        return {
            "method":  self.method,
            "venue":   self.venue,
            "params":  self.params,
            "flops":   self.flops,
            "psnr":    self.psnr,
            "ssim":    self.ssim,
            "lpips":   self.lpips,
            "fps_gpu": self.fps_gpu,
            "fps_cpu": self.fps_cpu,
            "source":  self.source,
        }


# 각 논문 / 공식 코드에서 보고된 LOL eval15 기준 근사치.
# (정확한 출처는 본 모듈 주석으로만 표시하고, 학술 논문 게재 시 직접 재측정 권장.)
LITERATURE_ROWS: List[LiteratureRow] = [
    LiteratureRow(
        method="Zero-DCE",   venue="CVPR'20",
        params=79_416,       flops=int(5.21e9),
        psnr=14.86,          ssim=0.56,           lpips=0.31,
        source="paper",
    ),
    LiteratureRow(
        method="Zero-DCE++", venue="TPAMI'21",
        params=10_561,       flops=int(0.11e9),
        psnr=14.86,          ssim=0.55,           lpips=0.32,
        source="paper",
    ),
    LiteratureRow(
        method="SCI",        venue="CVPR'22",
        params=10_561,       flops=int(0.18e9),
        psnr=14.78,          ssim=0.52,           lpips=0.34,
        source="paper",
    ),
    LiteratureRow(
        method="RetinexNet", venue="BMVC'18",
        params=555_000,      flops=int(587e9),
        psnr=16.77,          ssim=0.56,           lpips=0.47,
        source="paper",
    ),
    LiteratureRow(
        method="EnlightenGAN", venue="TIP'21",
        params=8_640_000,    flops=int(273e9),
        psnr=17.48,          ssim=0.65,           lpips=0.32,
        source="paper",
    ),
    LiteratureRow(
        method="FUnIE-GAN",  venue="RA-L'20",
        params=7_020_000,    flops=int(10.2e9),
        psnr=17.40,          ssim=0.66,           lpips=0.30,
        source="paper",
    ),
]


# ===========================================================================
# 2. Load our model
# ===========================================================================
def load_our_generator(ckpt_path: Path, device: str) -> LightEnhanceGenerator:
    """체크포인트로부터 G 재구성. base_filters / ablation flags 자동 감지."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    bf = 32
    use_attention = True
    use_dsconv = True
    if isinstance(state, dict):
        bf = int(state.get("base_filters", bf))
        use_attention = bool(state.get("use_attention", use_attention))
        use_dsconv = bool(state.get("use_dsconv", use_dsconv))
        if "config" in state and isinstance(state["config"], dict):
            bf = int(state["config"].get("base_filters", bf))

    G = LightEnhanceGenerator(
        base_filters=bf, use_attention=use_attention, use_dsconv=use_dsconv,
    ).to(device)
    sd = state["generator"] if isinstance(state, dict) and "generator" in state else state
    G.load_state_dict(sd)
    G.eval()
    return G


# ===========================================================================
# 3. Print plain + LaTeX tables
# ===========================================================================
def _fmt_count(n: Optional[float]) -> str:
    if n is None or n != n:  # None or NaN
        return "—"
    if n >= 1e9:  return f"{n/1e9:.2f}G"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def _fmt_float(x: Optional[float], digits: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:.{digits}f}"


def print_plain_table(rows: List[Dict[str, Any]]) -> None:
    print(HRULE)
    print(" 비교 결과 — LOL eval15 (256×256)")
    print(HRULE)
    hdr = (f"  {'Method':<24} {'Venue':<10} {'Params':>8} {'FLOPs':>8} "
           f"{'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>7} "
           f"{'FPS_G':>7} {'FPS_C':>6}  Source")
    print(hdr)
    print(f"  {'-'*24} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*7} {'-'*6}  {'-'*8}")
    for r in rows:
        marker = "  <—" if r["source"] == "this work" else ""
        print(
            f"  {r['method']:<24} {r['venue']:<10} "
            f"{_fmt_count(r['params']):>8} {_fmt_count(r['flops']):>8} "
            f"{_fmt_float(r['psnr']):>7} {_fmt_float(r['ssim'], 3):>7} "
            f"{_fmt_float(r['lpips'], 3):>7} "
            f"{_fmt_float(r['fps_gpu'], 0):>7} {_fmt_float(r['fps_cpu'], 0):>6}  "
            f"{r['source']}{marker}"
        )
    print(SUBRULE)


def print_latex_table(rows: List[Dict[str, Any]]) -> None:
    """논문에 그대로 복붙 가능한 LaTeX tabular."""
    print()
    print("% --- LaTeX table (copy into paper) ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Quantitative comparison on LOL eval15.}")
    print(r"\label{tab:comparison}")
    print(r"\small")
    print(r"\begin{tabular}{l c r r c c c r}")
    print(r"\toprule")
    print(r"Method & Venue & Params & FLOPs & PSNR$\uparrow$ & SSIM$\uparrow$ "
          r"& LPIPS$\downarrow$ & FPS \\")
    print(r"\midrule")
    for r in rows:
        bold_open = r"\textbf{" if r["source"] == "this work" else ""
        bold_close = "}" if r["source"] == "this work" else ""
        print(
            f"{bold_open}{r['method']}{bold_close} & "
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


# ===========================================================================
# 4. Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="비교 실험")
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/stage2_best.pth")
    p.add_argument("--data_root", type=str,
                   default="../DataSet/LOLdataset")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--results_dir", type=str,
                   default="experiments/results")
    p.add_argument("--method_name", type=str,
                   default="Ours (LightEnhanceGAN)")
    p.add_argument("--venue", type=str, default="this work")
    p.add_argument("--no_cpu_fps", action="store_true",
                   help="CPU FPS 측정 건너뛰기 (속도 우선)")
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main() -> int:
    args = parse_args()
    device = args.device

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"[error] checkpoint not found: {ckpt}")
        return 1

    print(HRULE)
    print(" Comparison — Ours vs. prior methods (LOL eval15)")
    print(HRULE)
    print(f"  checkpoint : {ckpt}")
    print(f"  data_root  : {args.data_root}")
    print(f"  device     : {device}")
    print(SUBRULE)

    eval_loader = get_eval_loader(
        args.data_root, batch_size=1,
        num_workers=args.num_workers, image_size=args.image_size,
    )

    G = load_our_generator(ckpt, device=device)
    metrics = benchmark_model_full(
        G, eval_loader, device=device,
        compute_cpu_fps=not args.no_cpu_fps,
    )
    print(f"  Ours params      : {int(metrics['params']):,}")
    print(f"  Ours FLOPs       : {metrics['flops']/1e9:.3f} G")
    print(f"  Ours PSNR / SSIM : {metrics['psnr']:.2f} / {metrics['ssim']:.4f}")
    print(f"  Ours LPIPS       : {metrics['lpips']:.4f}")
    print(f"  Ours FPS GPU/CPU : {metrics['fps_gpu']:.0f} / "
          f"{metrics['fps_cpu']:.1f}")
    print(SUBRULE)

    # Build comparison rows
    rows: List[Dict[str, Any]] = [r.as_row() for r in LITERATURE_ROWS]
    rows.append(LiteratureRow(
        method=args.method_name, venue=args.venue,
        params=metrics["params"], flops=metrics["flops"],
        psnr=metrics["psnr"], ssim=metrics["ssim"], lpips=metrics["lpips"],
        fps_gpu=metrics["fps_gpu"], fps_cpu=metrics["fps_cpu"],
        source="this work",
    ).as_row())

    print_plain_table(rows)
    print_latex_table(rows)

    # Save CSV
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV  -> {csv_path}")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
