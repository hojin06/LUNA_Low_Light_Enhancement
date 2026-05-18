"""λ_det sweep — train_detection_loss 를 여러 λ_det 값으로 순차 실행 후 비교.

목적 (Purpose)
--------------
``train_detection_loss.py`` 의 ``--lambda_det`` 값은 reconstruction vs.
detection 학습 신호의 trade-off 를 결정한다.  너무 작으면 mAP 향상이 미미,
너무 크면 PSNR/SSIM 이 무너진다.  본 sweep 스크립트는 ``--lambdas`` 리스트의
각 값을 동일 조건으로 학습하고 ExDark/LOL 평가로 *PSNR–mAP trade-off* 를
정량 비교하여 best balance 를 자동 추천한다.

동작 (How)
----------
1. 공유 리소스 (LoLI/LOL/ExDark loader, frozen YOLO + v8DetectionLoss,
   evaluation loader) 를 **한 번만** 빌드한다 — λ 별 재로딩 비용 절감.
2. 각 λ 마다:
   * 동일 ``--resume`` 체크포인트로부터 fresh start (``train_loop`` 내부의
     resume 로직 사용; 이전 λ 의 _last 체크포인트는 시작 전 삭제).
   * train_detection_loss 의 ``train_loop`` 그대로 호출 (per-λ epoch-level
     CSV / best_psnr 체크포인트는 정상 저장; per-epoch ExDark eval 은
     ``exdark_root=""`` 로 *비활성* — 본 sweep 이 final eval 한 번만 수행).
   * 학습 후 LoLI val / LOL eval15 / ExDark 100 sample mAP 최종 평가.
3. 모든 λ 결과를 비교표로 stdout 출력 + ``logs/lambda_sweep_results.csv``
   저장 + ``results/psnr_vs_map_tradeoff.png`` plot 자동 생성.
4. Best balance 자동 판정: ``LOL eval15 PSNR ≥ 19.0`` 인 row 들 중 mAP@0.5
   최대값을 갖는 λ → "★ 균형점" 표시.

실행 예
-------
.. code-block:: bash

    python lambda_sweep.py --lambdas 0.005 0.01 0.02 0.03 0.05 --epochs 15 \\
        --max_samples 5000 --resume checkpoints/loli_street_30000_stage2_best.pth \\
        --loli_root "..." --lol_root "..." --exdark_root "..."
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows 콘솔 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader

# 기존 인프라 재사용 (수정하지 않음)
from utils import evaluate
from utils.exdark_detection_dataset import (
    ExDarkDetectionDataset, exdark_yolo_collate,
)
from train_hybrid_v1_final import build_hybrid_v1_generator
from train_loli_street import (
    _measure_baseline_eval15, build_eval_loaders, build_mixed_train_loader,
)
from train_detection_loss import (
    DetLossCsvLogger, quick_exdark_eval, setup_frozen_yolo, train_loop,
    _parse_splits,
)


HRULE = "=" * 100
SUBRULE = "-" * 100


# ===========================================================================
# 1. 판정 기준 (논문 reporting 일관성을 위해 상수로 노출)
# ===========================================================================
BASELINE_PSNR: float = 20.95   # LoLI-30K 의 LOL eval15 PSNR
BASELINE_MAP:  float = 0.367   # LoLI-30K 의 ExDark 100-sample mAP@0.5
TARGET_MAP:    float = 0.447   # Original ExDark mAP@0.5 (논문 비교 기준)
BALANCE_PSNR_FLOOR: float = 19.0  # "PSNR ≥ 이 값" 이면 balance 후보


def _lambda_tag(lam: float) -> str:
    """0.005 → '0005', 0.01 → '0010', 0.05 → '0050' (×1000 4자리 zero-pad).

    train_detection_loss 의 tag 규칙 (×10 2자리) 보다 sweep 용은 더 세밀한 값
    까지 충돌 없이 표현하기 위해 ×1000 4자리.
    """
    return f"{int(round(lam * 1000)):04d}"


def _judge_row(
    psnr: float, map50: Optional[float],
    is_balance: bool,
) -> str:
    """LoLI-30K baseline 대비 한 줄 평. 표의 마지막 컬럼에 들어간다.

    규칙은 heuristic — 정확한 임계치는 위 BASELINE_* / BALANCE_PSNR_FLOOR
    상수 참조.
    """
    if is_balance:
        return "★ 균형점"
    if map50 is None:
        return "ExDark eval 실패"

    dpsnr = psnr - BASELINE_PSNR    # baseline 대비 변화량 (음수면 하락)
    dmap  = map50 - BASELINE_MAP

    if dpsnr < -3.0:
        return "PSNR 붕괴"
    if dpsnr < -2.0:
        return "PSNR 하락 주의"
    if dpsnr < -1.0:
        return "mAP↑ PSNR 허용범위" if dmap > 0.015 else "PSNR 하락 주의"
    if dpsnr < -0.5:
        if dmap > 0.015:
            return "PSNR 약간 하락, mAP↑"
        return "PSNR 약간 하락, mAP 미미"
    # PSNR within 0.5 of baseline
    if dmap > 0.015:
        return "PSNR 유지, mAP↑"
    if dmap > 0.005:
        return "PSNR 유지, mAP 약간↑"
    return "PSNR 유지, mAP 미미"


# ===========================================================================
# 2. 개별 λ 학습용 args namespace 합성
# ===========================================================================
def _build_lambda_args(
    sa: argparse.Namespace, lam: float, run_tag: str,
) -> argparse.Namespace:
    """sweep args + 특정 λ → train_detection_loss.train_loop 가 받는 Namespace."""
    return argparse.Namespace(
        # ---- 데이터 ----
        loli_root=sa.loli_root,
        lol_root=sa.lol_root,
        eval_lol_root=sa.eval_lol_root,
        exdark_root="",   # train_loop 내부 ExDark eval 차단 (sweep 이 최종 1회 수행)
        max_samples=sa.max_samples,
        mix_ratio=sa.mix_ratio,
        exdark_max_samples=sa.exdark_max_samples,
        exdark_train_splits=sa.exdark_train_splits,
        val_subset_size=sa.val_subset_size,
        image_size=sa.image_size,
        num_workers=sa.num_workers,
        batch_size=sa.batch_size,
        # ---- 학습 ----
        epochs=sa.epochs,
        lr=sa.lr,
        lambda_det=lam,
        warmup_epochs=sa.warmup_epochs,
        grad_clip_max=sa.grad_clip_max,
        lambda_l1=1.0, lambda_perceptual=0.5, lambda_ssim=1.0,
        beta1=0.9, beta2=0.999,
        weight_decay=sa.weight_decay,
        eta_min_ratio=0.1,
        # ---- YOLO ----
        yolo_weights=sa.yolo_weights,
        # ---- 평가 ----
        eval_every=1,
        eval_every_exdark=999,      # 실제 효력은 ``exdark_root=""`` 가 차단
        exdark_eval_samples=sa.exdark_eval_samples,
        exdark_conf=sa.exdark_conf,
        save_samples_every=999,     # train_loop 의 마지막 epoch 만 한 번 저장됨
        forget_threshold=19.0,
        # ---- I/O ----
        save_dir=sa.save_dir,
        log_path=str(sa.log_dir / f"det_loss_sweep_{run_tag}_epochs.csv"),
        samples_dir=str(sa.samples_dir / run_tag),
        resume=sa.resume,
        force=True,
        # ---- 런타임 ----
        device=sa.device,
        seed=sa.seed,
        run_tag=f"sweep_{run_tag}",
    )


# ===========================================================================
# 3. 비교표 (사용자 spec 의 양식 그대로)
# ===========================================================================
def _format_results_table(
    results: List[Dict[str, Any]], balance_tag: Optional[str],
) -> str:
    """sweep 결과 list → ASCII 표 문자열.  ``balance_tag`` 와 일치하는 row 에 ★."""
    lines: List[str] = []
    lines.append(HRULE)
    lines.append(" Lambda Sweep Results")
    lines.append(HRULE)
    lines.append(f"  {'λ_det':>7} | {'PSNR':>5} | {'SSIM':>5} | {'mAP@0.5':>7} | "
                 f"{'Recall':>6} | 판정")
    lines.append(SUBRULE)

    for r in results:
        is_balance = (r["tag"] == balance_tag)
        psnr = r["lol15_psnr"]
        ssim = r["lol15_ssim"]
        m50  = r["exdark_map50"]
        rec  = r["exdark_r"]
        m50_s = f"{m50:.4f}" if m50 is not None else "   —  "
        rec_s = f"{rec:.3f}" if rec is not None else "  —  "
        judgment = _judge_row(psnr, m50, is_balance)
        lines.append(
            f"  {r['lambda']:>7.4f} | {psnr:>5.2f} | {ssim:>5.3f} | "
            f"{m50_s:>7} | {rec_s:>6} | {judgment}"
        )

    lines.append(SUBRULE)
    if balance_tag is not None:
        # balance row 찾기
        br = next((x for x in results if x["tag"] == balance_tag), None)
        if br is not None:
            m50_s = f"{br['exdark_map50']:.4f}" if br["exdark_map50"] is not None else "—"
            lines.append(
                f"  Best balance: λ={br['lambda']}  "
                f"(LOL eval15 PSNR {br['lol15_psnr']:.2f}, mAP@0.5 {m50_s})"
            )
    else:
        lines.append("  Best balance: PSNR ≥ "
                     f"{BALANCE_PSNR_FLOOR} 인 후보가 없습니다.")
    lines.append(HRULE)
    return "\n".join(lines)


# ===========================================================================
# 4. CSV 저장
# ===========================================================================
SWEEP_CSV_FIELDS: List[str] = [
    "lambda", "tag",
    "loli_psnr", "loli_ssim",
    "lol15_psnr", "lol15_ssim",
    "exdark_map50", "exdark_map", "exdark_p", "exdark_r", "exdark_n_images",
    "epochs", "lr", "batch_size", "warmup_epochs",
    "is_balance", "judgment",
    "wall_time_min",
]


def _save_results_csv(
    results: List[Dict[str, Any]], balance_tag: Optional[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_CSV_FIELDS)
        w.writeheader()
        for r in results:
            is_bal = (r["tag"] == balance_tag)
            judgment = _judge_row(r["lol15_psnr"], r["exdark_map50"], is_bal)
            w.writerow({
                "lambda": r["lambda"], "tag": r["tag"],
                "loli_psnr": f"{r['loli_psnr']:.4f}",
                "loli_ssim": f"{r['loli_ssim']:.4f}",
                "lol15_psnr": f"{r['lol15_psnr']:.4f}",
                "lol15_ssim": f"{r['lol15_ssim']:.4f}",
                "exdark_map50":  f"{r['exdark_map50']:.6f}" if r['exdark_map50'] is not None else "",
                "exdark_map":    f"{r['exdark_map']:.6f}"   if r['exdark_map']   is not None else "",
                "exdark_p":      f"{r['exdark_p']:.4f}"     if r['exdark_p']     is not None else "",
                "exdark_r":      f"{r['exdark_r']:.4f}"     if r['exdark_r']     is not None else "",
                "exdark_n_images": r.get("exdark_n_images", ""),
                "epochs": r["epochs"], "lr": r["lr"],
                "batch_size": r["batch_size"], "warmup_epochs": r["warmup_epochs"],
                "is_balance": int(is_bal), "judgment": judgment,
                "wall_time_min": f"{r['wall_time_min']:.2f}",
            })


# ===========================================================================
# 5. PSNR vs mAP trade-off plot
# ===========================================================================
def _make_tradeoff_plot(
    results: List[Dict[str, Any]],
    balance_tag: Optional[str],
    out_path: Path,
) -> None:
    """X = LOL eval15 PSNR, Y = ExDark mAP@0.5.  점마다 λ 라벨 + baseline 점 + target line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib 미설치 — 그래프 생성 생략.")
        return

    # 유효 데이터 (ExDark eval 성공한 것만) 분리
    valid = [r for r in results if r["exdark_map50"] is not None]
    if not valid:
        print("  [plot] ExDark mAP 데이터 없음 — 그래프 생략.")
        return

    psnrs   = [r["lol15_psnr"]   for r in valid]
    maps50  = [r["exdark_map50"] for r in valid]
    lambdas = [r["lambda"]       for r in valid]
    tags    = [r["tag"]          for r in valid]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # ---- λ sweep 점 ----
    bal_idx: Optional[int] = None
    for k, t in enumerate(tags):
        if t == balance_tag:
            bal_idx = k; break
    sweep_color = "#1f77b4"
    ax.scatter(psnrs, maps50, s=70, c=sweep_color, edgecolors="white", linewidths=0.8,
               zorder=4, label=r"$\lambda_{det}$ sweep")
    if bal_idx is not None:
        ax.scatter([psnrs[bal_idx]], [maps50[bal_idx]],
                   s=260, marker="*", c="#d62728", edgecolors="black", linewidths=1.0,
                   zorder=5, label="Best balance (★)")
    for x, y, lam in zip(psnrs, maps50, lambdas):
        ax.annotate(
            rf"$\lambda$={lam}", (x, y),
            textcoords="offset points", xytext=(7, 7),
            fontsize=9, color=sweep_color, zorder=6,
        )

    # ---- LoLI-30K baseline 점 ----
    ax.scatter([BASELINE_PSNR], [BASELINE_MAP],
               s=140, marker="s", c="#2ca02c", edgecolors="black", linewidths=0.8,
               zorder=4, label=f"LoLI-30K baseline (no det loss)")
    ax.annotate(
        f"baseline\n({BASELINE_PSNR}, {BASELINE_MAP})",
        (BASELINE_PSNR, BASELINE_MAP),
        textcoords="offset points", xytext=(8, -22),
        fontsize=9, color="#2ca02c",
    )

    # ---- Target mAP line (Original ExDark) ----
    ax.axhline(TARGET_MAP, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Original ExDark mAP@0.5 = {TARGET_MAP}")

    ax.set_xlabel("LOL eval15 PSNR (dB)")
    ax.set_ylabel("ExDark 100-sample mAP@0.5")
    ax.set_title("LUNA Detection-Aware Loss: PSNR vs mAP Trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    # Axis padding
    all_x = psnrs + [BASELINE_PSNR]
    all_y = maps50 + [BASELINE_MAP, TARGET_MAP]
    x_pad = max(0.4, (max(all_x) - min(all_x)) * 0.15)
    y_pad = max(0.01, (max(all_y) - min(all_y)) * 0.15)
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


# ===========================================================================
# 6. CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="λ_det sweep — train_detection_loss 를 여러 λ 로 순차 실행 후 비교/plot",
    )
    # ---- sweep 전용 ----
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.005, 0.01, 0.02, 0.03, 0.05],
                   help="순차 실행할 λ_det 값들. 공백으로 구분.")
    p.add_argument("--epochs", type=int, default=15,
                   help="각 λ 의 학습 epoch 수. 빠른 스캔을 위해 train_detection_loss 의 20 보다 적게.")
    p.add_argument("--results_csv", type=str,
                   default="./logs/lambda_sweep_results.csv",
                   help="sweep 비교 결과 CSV 저장 경로.")
    p.add_argument("--plot_path", type=str,
                   default="./results/psnr_vs_map_tradeoff.png",
                   help="PSNR vs mAP trade-off 그래프 저장 경로.")
    p.add_argument("--per_lambda_log_dir", type=str, default="./logs/lambda_sweep",
                   help="각 λ 의 epoch-level CSV 가 들어갈 디렉토리.")
    p.add_argument("--sweep_samples_dir", type=str,
                   default="./results/lambda_sweep_samples")

    # ---- 데이터 (train_detection_loss 와 동일 옵션) ----
    p.add_argument("--loli_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LoLI-Street")
    p.add_argument("--lol_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LOL-v2")
    p.add_argument("--eval_lol_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\LOLdataset")
    p.add_argument("--exdark_root", type=str,
                   default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\ExDark")
    p.add_argument("--max_samples",        type=int, default=5000)
    p.add_argument("--mix_ratio",          type=float, default=0.2)
    p.add_argument("--exdark_max_samples", type=int, default=7363)
    p.add_argument("--exdark_train_splits", type=str, default="1,2,3")
    p.add_argument("--val_subset_size",    type=int, default=500)
    p.add_argument("--image_size",         type=int, default=256)
    p.add_argument("--num_workers",        type=int, default=4)
    p.add_argument("--batch_size",         type=int, default=2)

    # ---- 학습 ----
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--warmup_epochs", type=int,   default=3)
    p.add_argument("--grad_clip_max", type=float, default=1.0)
    p.add_argument("--weight_decay",  type=float, default=1e-4)

    # ---- YOLO / 평가 ----
    p.add_argument("--yolo_weights",        type=str,   default="yolov8n.pt")
    p.add_argument("--exdark_eval_samples", type=int,   default=100)
    p.add_argument("--exdark_conf",         type=float, default=0.25)

    # ---- I/O 공통 ----
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--resume",   type=str,
                   default="checkpoints/loli_street_30000_stage2_best.pth")

    # ---- 런타임 ----
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed",   type=int, default=42)

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if any(l < 0 for l in args.lambdas):
        raise ValueError(f"--lambdas 모두 ≥ 0 필요 (got {args.lambdas})")
    # 경로 정리
    args.results_csv = Path(args.results_csv).resolve()
    args.plot_path   = Path(args.plot_path).resolve()
    args.log_dir     = Path(args.per_lambda_log_dir).resolve()
    args.samples_dir = Path(args.sweep_samples_dir).resolve()
    return args


# ===========================================================================
# 7. Main
# ===========================================================================
def main() -> int:
    sa = parse_args()
    device = sa.device

    # ---- 저장 경로 사전 생성 ----
    Path(sa.save_dir).mkdir(parents=True, exist_ok=True)
    sa.log_dir.mkdir(parents=True, exist_ok=True)
    sa.samples_dir.mkdir(parents=True, exist_ok=True)
    sa.results_csv.parent.mkdir(parents=True, exist_ok=True)
    sa.plot_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(HRULE)
    print(" LUNA Lambda Sweep — λ_det = {} (epochs each = {}, lr = {})".format(
        sa.lambdas, sa.epochs, sa.lr,
    ))
    print(HRULE)
    print(f"  loli_root    : {sa.loli_root}")
    print(f"  lol_root     : {sa.lol_root}")
    print(f"  eval_lol_root: {sa.eval_lol_root}")
    print(f"  exdark_root  : {sa.exdark_root}")
    print(f"  resume       : {sa.resume}")
    print(f"  save_dir     : {sa.save_dir}")
    print(f"  log_dir      : {sa.log_dir}")
    print(f"  results_csv  : {sa.results_csv}")
    print(f"  plot_path    : {sa.plot_path}")
    print(f"  device       : {device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(SUBRULE)

    # ===================================================================
    # [1/4] 공유 리소스 빌드 (한 번)
    # ===================================================================
    print(" [1/4] Shared resources (loaders, frozen YOLO, baseline)")
    loli_loader, info = build_mixed_train_loader(
        loli_root=Path(sa.loli_root), lol_root=Path(sa.lol_root),
        max_samples=sa.max_samples, mix_ratio=sa.mix_ratio,
        image_size=sa.image_size, batch_size=sa.batch_size,
        num_workers=sa.num_workers, seed=sa.seed,
    )
    print(f"    LoLI/LOL mixed: {info['n_total']} samples, "
          f"batch={sa.batch_size}, steps≈{info['n_total']//max(sa.batch_size,1)}")

    splits_t = _parse_splits(sa.exdark_train_splits)
    if splits_t and 3 in splits_t:
        print("    [warn] ExDark training 에 split=3 (Test) 포함 — eval 과 overlap.")
    exdark_dataset = ExDarkDetectionDataset(
        exdark_root=Path(sa.exdark_root), image_size=sa.image_size,
        splits=splits_t, max_samples=sa.exdark_max_samples,
        filter_empty=True,
    )
    if len(exdark_dataset) == 0:
        print(f"[error] ExDark train sample 0 — exdark_root 확인: {sa.exdark_root}")
        return 1
    exdark_loader = DataLoader(
        exdark_dataset, batch_size=sa.batch_size, shuffle=True,
        num_workers=min(sa.num_workers, 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=min(sa.num_workers, 2) > 0,
        collate_fn=exdark_yolo_collate, drop_last=True,
    )
    print(f"    ExDark detection: {len(exdark_dataset)} samples "
          f"(splits = {splits_t or 'ALL'})")

    loli_val_loader, lol15_loader, n_val, n_e15 = build_eval_loaders(
        loli_root=Path(sa.loli_root),
        eval_lol_root=Path(sa.eval_lol_root),
        image_size=sa.image_size, num_workers=sa.num_workers,
        val_subset_size=sa.val_subset_size, seed=sa.seed,
    )
    print(f"    Eval loaders   : LoLI val={n_val}, LOL eval15={n_e15}")

    baseline_psnr = _measure_baseline_eval15(
        Path(sa.resume), lol15_loader, device=device,
    )
    if baseline_psnr is not None:
        print(f"    Resume baseline LOL eval15 PSNR = {baseline_psnr:.3f}")

    print(" [2/4] Frozen YOLOv8n + v8DetectionLoss")
    G = build_hybrid_v1_generator().to(device)
    yolo_model, det_loss_fn, yolo_predict_model = setup_frozen_yolo(
        weights=sa.yolo_weights, device=device,
    )
    g_n = sum(p.numel() for p in G.parameters())
    y_n_train = sum(p.numel() for p in yolo_model.parameters() if p.requires_grad)
    print(f"    G params={g_n:,} | YOLO trainable params={y_n_train}  "
          f"(must be 0)")
    assert y_n_train == 0

    # ===================================================================
    # [3/4] λ sweep loop
    # ===================================================================
    print(" [3/4] λ sweep — {} runs × {} epochs".format(len(sa.lambdas), sa.epochs))
    print(SUBRULE)
    results: List[Dict[str, Any]] = []
    overall_t0 = time.perf_counter()

    for li, lam in enumerate(sa.lambdas, start=1):
        run_tag = _lambda_tag(lam)
        print()
        print(HRULE)
        print(f"  Sweep [{li}/{len(sa.lambdas)}]   λ_det = {lam}   tag = {run_tag}")
        print(HRULE)

        per_args = _build_lambda_args(sa, lam, run_tag)

        # per-λ paths (train_loop 가 best_psnr 만 저장; best_map 은 어차피 미발생)
        per_paths: Dict[str, Path] = {
            "best_psnr":   Path(sa.save_dir) / f"det_loss_sweep_{run_tag}_best_psnr.pth",
            "best_map":    Path(sa.save_dir) / f"det_loss_sweep_{run_tag}_best_map.pth",
            "last":        Path(sa.save_dir) / f"det_loss_sweep_{run_tag}_last.pth",
            "samples_dir": sa.samples_dir / run_tag,
        }
        # Fresh start — 이전 sweep run 의 잔재 삭제
        for k in ("best_psnr", "best_map", "last"):
            if per_paths[k].exists():
                per_paths[k].unlink()
        per_paths["samples_dir"].mkdir(parents=True, exist_ok=True)

        # per-λ epoch-level CSV
        per_csv_path = sa.log_dir / f"det_loss_sweep_{run_tag}_epochs.csv"
        per_logger = DetLossCsvLogger(per_csv_path)

        # ---- 학습 (train_detection_loss.train_loop 그대로) ----
        t0 = time.perf_counter()
        train_loop(
            args=per_args, paths=per_paths, G=G,
            yolo_model=yolo_model, det_loss_fn=det_loss_fn,
            yolo_predict_model=yolo_predict_model,
            loli_loader=loli_loader, exdark_loader=exdark_loader,
            loli_val_loader=loli_val_loader, lol15_loader=lol15_loader,
            baseline_eval15_psnr=baseline_psnr,
            csv_logger=per_logger,
        )
        wall_min = (time.perf_counter() - t0) / 60.0

        # ---- 최종 평가 (G 는 마지막 epoch 상태) ----
        print()
        print(f"  [final eval] λ={lam}")
        loli_v  = evaluate(G, loli_val_loader, device=device)
        lol15_v = evaluate(G, lol15_loader,   device=device)
        ex_v = quick_exdark_eval(
            G=G, exdark_root=Path(sa.exdark_root),
            yolo_predict_model=yolo_predict_model,
            image_size=sa.image_size, device=device,
            num_samples=sa.exdark_eval_samples,
            conf=sa.exdark_conf, seed=sa.seed,
        )

        row: Dict[str, Any] = {
            "lambda": lam, "tag": run_tag,
            "loli_psnr": float(loli_v["psnr"]),
            "loli_ssim": float(loli_v["ssim"]),
            "lol15_psnr": float(lol15_v["psnr"]),
            "lol15_ssim": float(lol15_v["ssim"]),
            "exdark_map50":  ex_v["map50"]    if ex_v else None,
            "exdark_map":    ex_v["map"]      if ex_v else None,
            "exdark_p":      ex_v["p"]        if ex_v else None,
            "exdark_r":      ex_v["r"]        if ex_v else None,
            "exdark_n_images": ex_v["n_images"] if ex_v else None,
            "epochs": sa.epochs, "lr": sa.lr,
            "batch_size": sa.batch_size, "warmup_epochs": sa.warmup_epochs,
            "wall_time_min": wall_min,
        }
        ex_str = f"mAP@0.5={ex_v['map50']:.4f}" if ex_v else "ExDark eval failed"
        print(f"    LoLI={loli_v['psnr']:.2f}/{loli_v['ssim']:.4f} | "
              f"e15={lol15_v['psnr']:.2f}/{lol15_v['ssim']:.4f} | "
              f"{ex_str} | wall={wall_min:.1f} min")
        results.append(row)

    total_min = (time.perf_counter() - overall_t0) / 60.0

    # ===================================================================
    # [4/4] Best balance 판정 + 표 + CSV + plot
    # ===================================================================
    # PSNR >= BALANCE_PSNR_FLOOR 인 row 중 mAP@0.5 최대값.  하나도 없으면 None.
    candidates = [
        r for r in results
        if r["exdark_map50"] is not None and r["lol15_psnr"] >= BALANCE_PSNR_FLOOR
    ]
    balance_tag: Optional[str] = None
    if candidates:
        best = max(candidates, key=lambda r: r["exdark_map50"])
        balance_tag = best["tag"]

    print()
    print(_format_results_table(results, balance_tag))
    print()
    print(f"  Total sweep wall time: {total_min:.1f} min")

    _save_results_csv(results, balance_tag, sa.results_csv)
    print(f"  Results CSV  → {sa.results_csv}")

    _make_tradeoff_plot(results, balance_tag, sa.plot_path)
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
