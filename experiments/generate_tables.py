"""실험 결과 종합 — 콘솔 표 / LaTeX 표 / matplotlib 그래프 생성.

입력 (있는 만큼 읽음, 없는 것은 건너뜀)
-----------------------------------------
* ``experiments/results/ablation_summary.csv``       — variant 별 종합 지표
* ``experiments/results/ablation_<name>_log.csv``    — variant 별 epoch log
* ``experiments/results/comparison_table.csv``       — 우리 vs prior methods
* ``logs/train_log_stage1.csv`` (옵션)               — 본 학습 epoch log
* ``logs/train_log_stage2.csv`` (옵션)

출력
----
* stdout : Ablation / Comparison 표 + LaTeX 버전
* ``experiments/results/plot_psnr_vs_epoch.png``    — Ablation 곡선
* ``experiments/results/plot_loss_vs_epoch.png``    — Ablation 곡선
* ``experiments/results/plot_ablation_bars.png``    — PSNR / SSIM / LPIPS 막대
* ``experiments/results/plot_stage_psnr.png`` (옵션) — Stage1→Stage2 본학습 곡선

사용법
------
.. code-block:: bash

    python experiments/generate_tables.py
"""
from __future__ import annotations

import argparse
import csv
import sys
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

import matplotlib
matplotlib.use("Agg")  # headless 환경 안전
import matplotlib.pyplot as plt


HRULE = "=" * 100
SUBRULE = "-" * 100


# ===========================================================================
# Helpers
# ===========================================================================
def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(s: Optional[str]) -> float:
    if s is None or s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _fmt_count(n: float) -> str:
    if n != n:
        return "—"
    if n >= 1e9:  return f"{n/1e9:.2f}G"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def _fmt(x: float, digits: int = 3) -> str:
    if x != x:
        return "—"
    return f"{x:.{digits}f}"


# ===========================================================================
# Ablation table
# ===========================================================================
def print_ablation_tables(summary_rows: List[Dict[str, str]]) -> None:
    if not summary_rows:
        print("[ablation] ablation_summary.csv 가 없습니다.  먼저 ablation.py 실행.")
        return

    print(HRULE)
    print(" Ablation Study — 결과")
    print(HRULE)
    hdr = (f"  {'Variant':<18} {'Params':>8} {'FLOPs':>8} "
           f"{'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>7} {'FPS_G':>7}  Notes")
    print(hdr)
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*32}")
    for r in summary_rows:
        print(
            f"  {r['name']:<18} "
            f"{_fmt_count(_to_float(r['params'])):>8} "
            f"{_fmt_count(_to_float(r['flops'])):>8} "
            f"{_fmt(_to_float(r['psnr']), 2):>7} "
            f"{_fmt(_to_float(r['ssim']), 3):>7} "
            f"{_fmt(_to_float(r['lpips']), 3):>7} "
            f"{_fmt(_to_float(r['fps_gpu']), 0):>7}  "
            f"{r.get('description', '')[:32]}"
        )
    print(SUBRULE)

    # LaTeX
    print()
    print("% --- Ablation LaTeX table ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study on LOL eval15.  All variants trained "
          r"with Stage~1 (supervised) for the same number of epochs.}")
    print(r"\label{tab:ablation}")
    print(r"\small")
    print(r"\begin{tabular}{l r r c c c r}")
    print(r"\toprule")
    print(r"Variant & Params & FLOPs & PSNR$\uparrow$ & SSIM$\uparrow$ "
          r"& LPIPS$\downarrow$ & FPS \\")
    print(r"\midrule")
    for r in summary_rows:
        is_full = r["name"] == "full"
        bo = r"\textbf{" if is_full else ""
        bc = "}" if is_full else ""
        # variant name 의 underscore → LaTeX 안전화
        v_name = r["name"].replace("_", r"\_")
        print(
            f"{bo}{v_name}{bc} & "
            f"{_fmt_count(_to_float(r['params']))} & "
            f"{_fmt_count(_to_float(r['flops']))} & "
            f"{_fmt(_to_float(r['psnr']), 2)} & "
            f"{_fmt(_to_float(r['ssim']), 3)} & "
            f"{_fmt(_to_float(r['lpips']), 3)} & "
            f"{_fmt(_to_float(r['fps_gpu']), 0)} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("% --- end Ablation LaTeX ---")
    print()


# ===========================================================================
# Comparison table — already printed by comparison.py; re-print here
# ===========================================================================
def print_comparison_table(rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("[comparison] comparison_table.csv 가 없습니다.  "
              "comparison.py 를 먼저 실행하세요.")
        return
    print(HRULE)
    print(" Comparison — Ours vs. prior methods (LOL eval15)")
    print(HRULE)
    hdr = (f"  {'Method':<24} {'Venue':<10} {'Params':>8} {'FLOPs':>8} "
           f"{'PSNR↑':>7} {'SSIM↑':>7} {'LPIPS↓':>7} {'FPS_G':>7}  Source")
    print(hdr)
    print(f"  {'-'*24} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*7}  {'-'*8}")
    for r in rows:
        marker = "  <—" if r.get("source") == "this work" else ""
        print(
            f"  {r['method']:<24} {r.get('venue',''):<10} "
            f"{_fmt_count(_to_float(r['params'])):>8} "
            f"{_fmt_count(_to_float(r['flops'])):>8} "
            f"{_fmt(_to_float(r['psnr']), 2):>7} "
            f"{_fmt(_to_float(r['ssim']), 3):>7} "
            f"{_fmt(_to_float(r['lpips']), 3):>7} "
            f"{_fmt(_to_float(r['fps_gpu']), 0):>7}  "
            f"{r.get('source','')}{marker}"
        )
    print(SUBRULE)


# ===========================================================================
# Plots
# ===========================================================================
def plot_ablation_curves(results_dir: Path) -> None:
    """ablation_<name>_log.csv 들을 모아 PSNR / Loss vs epoch 그래프."""
    logs = sorted(results_dir.glob("ablation_*_log.csv"))
    if not logs:
        print("[plot] ablation_*_log.csv 가 없어 곡선 생성 건너뜀.")
        return

    # ---- PSNR ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in logs:
        name = p.stem.replace("ablation_", "").replace("_log", "")
        rows = _read_csv(p)
        if not rows:
            continue
        epochs = [_to_float(r["epoch"]) for r in rows]
        eval_psnr = [_to_float(r["eval_psnr"]) for r in rows]
        ax.plot(epochs, eval_psnr, marker="o", markersize=3,
                linewidth=1.5, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eval PSNR (dB)")
    ax.set_title("Ablation — Eval PSNR vs. Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = results_dir / "plot_psnr_vs_epoch.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")

    # ---- Loss (total) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in logs:
        name = p.stem.replace("ablation_", "").replace("_log", "")
        rows = _read_csv(p)
        if not rows:
            continue
        epochs = [_to_float(r["epoch"]) for r in rows]
        g_loss = [_to_float(r["g_loss"]) for r in rows]
        ax.plot(epochs, g_loss, marker="o", markersize=3,
                linewidth=1.5, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("G Loss (L1 + λ·VGG + λ·SSIM)")
    ax.set_title("Ablation — Training Loss vs. Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = results_dir / "plot_loss_vs_epoch.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_ablation_bars(results_dir: Path,
                       summary_rows: List[Dict[str, str]]) -> None:
    """PSNR / SSIM / LPIPS 막대 그래프 (variant 별)."""
    if not summary_rows:
        return
    names  = [r["name"] for r in summary_rows]
    psnr   = [_to_float(r["psnr"])  for r in summary_rows]
    ssim   = [_to_float(r["ssim"])  for r in summary_rows]
    lpips_ = [_to_float(r["lpips"]) for r in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, title, fmt in [
        (axes[0], psnr,   "PSNR↑",  "%.2f"),
        (axes[1], ssim,   "SSIM↑",  "%.3f"),
        (axes[2], lpips_, "LPIPS↓", "%.3f"),
    ]:
        bars = ax.bar(names, vals, color=["#3a7bd5", "#888", "#888", "#888", "#888"])
        # `full` 강조
        for i, n in enumerate(names):
            if n == "full":
                bars[i].set_color("#d04848")
        ax.set_title(title)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, fmt % v, ha="center", va="bottom", fontsize=7)
    fig.suptitle("Ablation — quantitative metrics on LOL eval15")
    fig.tight_layout()
    out = results_dir / "plot_ablation_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_stage_curves(logs_dir: Path, results_dir: Path) -> None:
    """logs/ 의 Stage1 / Stage2 train_log csv 가 있으면 본 학습 PSNR 곡선."""
    s1 = logs_dir / "train_log_stage1.csv"
    s2 = logs_dir / "train_log_stage2.csv"
    if not s1.is_file() and not s2.is_file():
        print("[plot] logs/train_log_stage{1,2}.csv 없음 — stage 곡선 건너뜀.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    def _load(path: Path, label: str, color: str) -> None:
        rows = _read_csv(path)
        if not rows:
            return
        eps = [_to_float(r["epoch"]) for r in rows]
        ps = [_to_float(r["eval_psnr"]) for r in rows]
        ax.plot(eps, ps, marker="o", markersize=3, linewidth=1.5,
                label=label, color=color)

    _load(s1, "Stage 1 (supervised)", "#3a7bd5")
    if s2.is_file():
        # Stage 2 의 epoch 을 Stage 1 마지막 뒤에 잇기
        rows1 = _read_csv(s1)
        offset = float(rows1[-1]["epoch"]) if rows1 else 0.0
        rows = _read_csv(s2)
        if rows:
            eps = [_to_float(r["epoch"]) + offset for r in rows]
            ps = [_to_float(r["eval_psnr"]) for r in rows]
            ax.plot(eps, ps, marker="s", markersize=3, linewidth=1.5,
                    label="Stage 2 (GAN fine-tune)", color="#d04848")
            ax.axvline(x=offset, linestyle="--", color="gray", alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eval PSNR (dB)")
    ax.set_title("Two-stage training — Eval PSNR")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = results_dir / "plot_stage_psnr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ===========================================================================
# Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="결과 표 / 그래프 생성")
    p.add_argument("--results_dir", type=str,
                   default="experiments/results")
    p.add_argument("--logs_dir", type=str, default="logs",
                   help="본 학습 train_log_stage{1,2}.csv 가 있는 폴더")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _read_csv(results_dir / "ablation_summary.csv")
    comp_rows    = _read_csv(results_dir / "comparison_table.csv")

    # ---- Tables ----
    print_ablation_tables(summary_rows)
    print_comparison_table(comp_rows)

    # ---- Plots ----
    print(HRULE)
    print(" Generating plots")
    print(HRULE)
    plot_ablation_curves(results_dir)
    plot_ablation_bars(results_dir, summary_rows)
    plot_stage_curves(logs_dir, results_dir)
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
