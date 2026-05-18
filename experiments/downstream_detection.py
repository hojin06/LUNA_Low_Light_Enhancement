"""Downstream Task 실험 — YOLOv8 객체 검출로 LUNA 향상 효과 검증.

목적 (Purpose)
--------------
LUNA(LightEnhanceGenerator) 가 PSNR/SSIM 같은 *pixel-level* 지표 외에
**downstream computer vision task** 에서도 실효를 내는지 정량/정성 평가한다.
LOL eval15 의 (low, high) 페어와 LUNA 가 만든 enhanced 이미지에 동일한
YOLOv8n(pre-trained on MS-COCO) 모델을 추론만으로 적용하여 세 가지의 검출
결과를 비교한다.

실험 절차 (Pipeline)
--------------------
1. LOL eval15 의 각 페어를 (low, high) PIL 로 로드 → 256×256 BILINEAR resize.
2. low 를 LUNA 에 입력 → enhanced 텐서 → uint8 RGB 로 복원.
3. 세 이미지를 모두 YOLOv8n 으로 추론 (conf_threshold = 0.25).
4. 검출 결과에 대해
   * 총 검출 수 (total detections)
   * 평균 confidence
   * 검출된 클래스 종류 수 (unique class count)
   를 집계.
5. 시각 비교 PNG ``[Low+bbox | Enhanced+bbox | GT+bbox]`` 가로 배치 저장.
6. 이미지별 + 전체 평균을 CSV 로 저장.

본 스크립트는 학습/평가 코드를 일체 수정하지 않는다. LUNA 가중치 로드 시
``train_hybrid_v1_final.HYBRID_V1_CONV_CONFIG`` 와 동일한 hybrid_v1 사양으로
generator 를 재구성하여 ``ext_lol_v2_real_stage2_best.pth`` 와 호환되도록 한다.

사용법
------
.. code-block:: bash

    pip install ultralytics
    python experiments/downstream_detection.py \\
        --data_root "C:\\대학교\\Projects\\SmallSizePM_GAN_model\\DataSet\\LOLdataset" \\
        --checkpoint checkpoints/ext_lol_v2_real_stage2_best.pth
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- 프로젝트 루트를 sys.path 에 등록 (models/ 등 import 용) ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Windows 콘솔(cp949) 한글 출력 안전
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from models import LightEnhanceGenerator
# hybrid_v1 사양은 학습 스크립트의 상수를 그대로 재사용 (절대 변경 금지).
from train_hybrid_v1_final import (
    HYBRID_V1_BASE_FILTERS,
    HYBRID_V1_CONV_CONFIG,
    HYBRID_V1_USE_ATTENTION,
)


HRULE = "=" * 96
SUBRULE = "-" * 96


# ===========================================================================
# 1. LUNA Generator 로드 — hybrid_v1 사양 (input_conv=standard, 나머지 dsconv)
# ===========================================================================
def load_luna_generator(ckpt_path: Path, device: str) -> LightEnhanceGenerator:
    """체크포인트로부터 hybrid_v1 LUNA generator 재구성 + weight 로드.

    체크포인트 내부에 ``conv_config`` / ``base_filters`` 가 있으면 그 값을 우선
    사용 (메타데이터 일관성). 없으면 ``train_hybrid_v1_final`` 의 상수를 사용.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 메타데이터 우선 (체크포인트가 학습 시 함께 저장한 것)
    bf = HYBRID_V1_BASE_FILTERS
    use_attn = HYBRID_V1_USE_ATTENTION
    conv_cfg: Dict[str, str] = HYBRID_V1_CONV_CONFIG.copy()
    if isinstance(state, dict):
        bf = int(state.get("base_filters", bf))
        use_attn = bool(state.get("use_attention", use_attn))
        if "conv_config" in state and isinstance(state["conv_config"], dict):
            conv_cfg = dict(state["conv_config"])

    G = LightEnhanceGenerator(
        base_filters=bf,
        use_attention=use_attn,
        conv_config=conv_cfg,
    ).to(device)

    sd = state["generator"] if isinstance(state, dict) and "generator" in state else state
    G.load_state_dict(sd)
    G.eval()
    return G


# ===========================================================================
# 2. LOL eval15 페어 수집
# ===========================================================================
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def collect_eval15_pairs(data_root: Path) -> List[Tuple[Path, Path]]:
    """``LOLdataset/eval15/{low,high}`` 에서 파일명 일치 (low, high) 페어 목록.

    학습 코드의 ``data/dataset.py:_build_pairs`` 와 동일한 매칭 규칙.
    """
    eval_dir = data_root / "eval15"
    low_dir = eval_dir / "low"
    high_dir = eval_dir / "high"
    if not low_dir.is_dir() or not high_dir.is_dir():
        raise FileNotFoundError(
            f"LOL eval15 폴더를 찾을 수 없습니다:\n"
            f"  low  = {low_dir}\n  high = {high_dir}"
        )
    high_index = {
        p.stem: p for p in high_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    }
    pairs: List[Tuple[Path, Path]] = []
    for low_p in sorted(low_dir.iterdir()):
        if not low_p.is_file() or low_p.suffix.lower() not in _IMG_EXTS:
            continue
        high_p = high_index.get(low_p.stem)
        if high_p is not None:
            pairs.append((low_p, high_p))
    if not pairs:
        raise RuntimeError(f"eval15 (low, high) 페어가 0 개입니다: {eval_dir}")
    return pairs


# ===========================================================================
# 3. 이미지 ↔ Tensor 변환 헬퍼
# ===========================================================================
def pil_to_norm_tensor(img: Image.Image, image_size: int) -> torch.Tensor:
    """PIL RGB → 256×256 BILINEAR resize → tensor [-1, 1] (1, 3, H, W).

    학습 시 ``PairedAugment._eval_path`` 와 동일한 전처리 규칙을 따른다.
    """
    img = TF.resize(img, [image_size, image_size],
                    interpolation=InterpolationMode.BILINEAR)
    t = TF.to_tensor(img)            # [0, 1]
    t = t * 2.0 - 1.0                 # [-1, 1]
    return t.unsqueeze(0)            # (1, 3, H, W)


def norm_tensor_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """generator 출력 (값 범위 [-1, 1], (1, 3, H, W)) → uint8 RGB (H, W, 3).

    1) clamp([-1, 1])  2) [-1,1] → [0,1]  3) ×255 → uint8.
    """
    t = t.detach().clamp(-1.0, 1.0)
    t = (t + 1.0) * 0.5                          # [0, 1]
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    return arr  # RGB


def pil_to_uint8_rgb(img: Image.Image, image_size: int) -> np.ndarray:
    """PIL → 256×256 resize → uint8 RGB ndarray (시각화/YOLO 입력 공용)."""
    img = TF.resize(img, [image_size, image_size],
                    interpolation=InterpolationMode.BILINEAR)
    return np.array(img.convert("RGB"), dtype=np.uint8)


# ===========================================================================
# 4. YOLOv8 추론 + 결과 집계
# ===========================================================================
def _run_yolo_on_rgb(model, rgb: np.ndarray, conf: float, device: str):
    """YOLOv8 모델에 RGB uint8 ndarray 입력 → ultralytics ``Results`` 1개 반환.

    ultralytics 는 numpy 입력 시 BGR 을 가정하므로 RGB→BGR 변환 후 전달.
    """
    bgr = rgb[..., ::-1].copy()       # RGB → BGR (cv2 convention)
    results = model.predict(
        bgr,
        conf=conf,
        verbose=False,
        device=device,
    )
    return results[0]


def _summarize_result(result) -> Dict[str, Any]:
    """단일 YOLO ``Results`` → (n_det, mean_conf, unique_classes, names)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "n_det":         0,
            "mean_conf":     float("nan"),
            "n_classes":     0,
            "class_names":   [],
        }
    confs = boxes.conf.detach().cpu().numpy().astype(float)
    clses = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    names_map = result.names  # {idx: name}
    unique_ids = sorted(set(clses))
    return {
        "n_det":       int(len(boxes)),
        "mean_conf":   float(confs.mean()) if confs.size else float("nan"),
        "n_classes":   int(len(unique_ids)),
        "class_names": [names_map.get(i, str(i)) for i in unique_ids],
    }


# ===========================================================================
# 5. 시각화 — bbox 그려진 세 이미지 가로 배치 PNG 저장
# ===========================================================================
def _annotated_rgb(result) -> np.ndarray:
    """YOLO ``Results.plot()`` 의 BGR 출력을 RGB uint8 로 변환."""
    bgr = result.plot()  # (H, W, 3) uint8, BGR
    return bgr[..., ::-1].copy()


def save_comparison_triplet(
    low_annot: np.ndarray,
    enh_annot: np.ndarray,
    gt_annot:  np.ndarray,
    out_path: Path,
    pad: int = 6,
) -> None:
    """[Low+bbox | Enhanced+bbox | GT+bbox] 가로 배치 저장.

    셋이 모두 동일 해상도 (H, W, 3) uint8 RGB 라고 가정. 각 패널 사이에
    ``pad`` 픽셀의 흰색 separator 를 둔다.
    """
    h, w, _ = low_annot.shape
    sep = np.full((h, pad, 3), 255, dtype=np.uint8)
    concat = np.concatenate([low_annot, sep, enh_annot, sep, gt_annot], axis=1)

    # 상단에 label 띠 추가 (단순 흰색 + 검정 텍스트). 폰트 없이 PIL 기본 사용.
    label_h = 22
    label_bar = np.full((label_h, concat.shape[1], 3), 255, dtype=np.uint8)
    img = np.concatenate([label_bar, concat], axis=0)

    pil = Image.fromarray(img)
    from PIL import ImageDraw  # 지역 import — matplotlib 의존 없이 가벼움
    draw = ImageDraw.Draw(pil)
    # 각 패널 중앙에 라벨
    centers_x = [w // 2, w + pad + w // 2, 2 * (w + pad) + w // 2]
    for cx, label in zip(centers_x, ("Low (input)", "Enhanced (LUNA)", "GT (high)")):
        # 텍스트 폭 추정용 textbbox (Pillow ≥ 8)
        try:
            l, t, r, b = draw.textbbox((0, 0), label)
            tw = r - l
        except Exception:
            tw = 8 * len(label)
        draw.text((cx - tw // 2, 4), label, fill=(0, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


# ===========================================================================
# 6. 출력 — 비교표
# ===========================================================================
def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and x != x):  # NaN
        return "—"
    return f"{x:.{digits}f}"


def print_comparison_table(rows: List[Dict[str, Any]],
                           overall: Dict[str, Dict[str, float]]) -> None:
    """이미지별 + 전체 평균 비교표를 stdout 에 출력."""
    print(HRULE)
    print(" YOLOv8 Downstream Detection — Low vs. Enhanced (LUNA) vs. GT (high)")
    print(HRULE)
    hdr = (f"  {'Image':<14}"
           f" | {'Low (det / conf / cls)':>26}"
           f" | {'Enhanced (det / conf / cls)':>30}"
           f" | {'GT (det / conf / cls)':>26}")
    print(hdr)
    print(SUBRULE)
    for r in rows:
        def _cell(prefix: str) -> str:
            return (f"{r[f'{prefix}_n_det']:>3} / "
                    f"{_fmt_float(r[f'{prefix}_mean_conf'], 3):>5} / "
                    f"{r[f'{prefix}_n_classes']:>2}")
        print(f"  {r['image']:<14}"
              f" | {_cell('low'):>26}"
              f" | {_cell('enh'):>30}"
              f" | {_cell('gt'):>26}")
    print(SUBRULE)
    # 전체 평균 행
    def _avg_cell(d: Dict[str, float]) -> str:
        return (f"{_fmt_float(d['n_det'], 2):>5} / "
                f"{_fmt_float(d['mean_conf'], 3):>5} / "
                f"{_fmt_float(d['n_classes'], 2):>4}")
    print(f"  {'AVG (over N)':<14}"
          f" | {_avg_cell(overall['low']):>26}"
          f" | {_avg_cell(overall['enh']):>30}"
          f" | {_avg_cell(overall['gt']):>26}")
    print(HRULE)
    # 단일줄 요약 (논문 본문용)
    delta_det = overall["enh"]["n_det"] - overall["low"]["n_det"]
    delta_conf = overall["enh"]["mean_conf"] - overall["low"]["mean_conf"]
    print(f"  ΔDetections (Enhanced − Low) : {delta_det:+.2f}")
    print(f"  ΔMean confidence (Enh − Low) : {delta_conf:+.3f}")
    print(HRULE)


# ===========================================================================
# 7. Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLOv8 downstream detection — LUNA 향상 효과 검증",
    )
    p.add_argument("--data_root", type=str, required=True,
                   help="LOLdataset 폴더 (eval15/ 포함)")
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/ext_lol_v2_real_stage2_best.pth",
                   help="LUNA generator weight (hybrid_v1 호환)")
    p.add_argument("--yolo_weights", type=str, default="yolov8n.pt",
                   help="YOLOv8 가중치. 없으면 ultralytics 가 자동 다운로드.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="YOLO confidence threshold")
    p.add_argument("--image_size", type=int, default=256,
                   help="YOLO 추론 해상도 (LOL 학습/평가와 동일)")
    p.add_argument("--results_dir", type=str,
                   default="experiments/results/downstream",
                   help="시각 비교 PNG / CSV 저장 디렉토리")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (기본: cuda 가용 시 cuda)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 이면 전체 페어, 양수면 앞에서 N 개만 사용 (디버그)")
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main() -> int:
    args = parse_args()
    device = args.device

    # --- ultralytics import 는 main 안에서 (설치 안내 출력 가능) ---
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics 가 설치되어 있지 않습니다.")
        print("        pip install ultralytics")
        return 1

    data_root = Path(args.data_root)
    ckpt_path = Path(args.checkpoint)
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        print(f"[error] LUNA checkpoint not found: {ckpt_path}")
        return 1

    print(HRULE)
    print(" Downstream Detection (YOLOv8n on LOL eval15)")
    print(HRULE)
    print(f"  data_root    : {data_root}")
    print(f"  checkpoint   : {ckpt_path}")
    print(f"  yolo_weights : {args.yolo_weights}")
    print(f"  conf_thresh  : {args.conf}")
    print(f"  image_size   : {args.image_size}")
    print(f"  device       : {device}")
    print(f"  results_dir  : {results_dir}")
    print(SUBRULE)

    # ---- LUNA generator ----
    G = load_luna_generator(ckpt_path, device=device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"  LUNA params  : {n_params:,}  ({n_params/1e3:.1f} K)")

    # ---- YOLOv8 ----
    print(f"  YOLO loading : {args.yolo_weights} ...")
    yolo = YOLO(args.yolo_weights)  # 없으면 자동 다운로드
    print(f"  YOLO ready   : {len(yolo.names)} classes (COCO pre-trained)")
    print(SUBRULE)

    # ---- 페어 수집 ----
    pairs = collect_eval15_pairs(data_root)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    print(f"  pairs        : {len(pairs)}")
    print(SUBRULE)

    # ---- 순회 + 추론 ----
    rows: List[Dict[str, Any]] = []
    sums = {
        "low": {"n_det": 0.0, "mean_conf": 0.0, "n_classes": 0.0,
                "conf_weight": 0},
        "enh": {"n_det": 0.0, "mean_conf": 0.0, "n_classes": 0.0,
                "conf_weight": 0},
        "gt":  {"n_det": 0.0, "mean_conf": 0.0, "n_classes": 0.0,
                "conf_weight": 0},
    }

    with torch.no_grad():
        for low_p, high_p in pairs:
            stem = low_p.stem
            low_pil  = Image.open(low_p).convert("RGB")
            high_pil = Image.open(high_p).convert("RGB")

            # --- 입력 준비 ---
            low_rgb = pil_to_uint8_rgb(low_pil,  args.image_size)
            gt_rgb  = pil_to_uint8_rgb(high_pil, args.image_size)

            # LUNA 향상
            low_norm = pil_to_norm_tensor(low_pil, args.image_size).to(device)
            enh_norm = G(low_norm)
            enh_rgb = norm_tensor_to_uint8_rgb(enh_norm)

            # --- YOLO 추론 ---
            r_low = _run_yolo_on_rgb(yolo, low_rgb, args.conf, device)
            r_enh = _run_yolo_on_rgb(yolo, enh_rgb, args.conf, device)
            r_gt  = _run_yolo_on_rgb(yolo, gt_rgb,  args.conf, device)

            # --- 결과 집계 ---
            s_low = _summarize_result(r_low)
            s_enh = _summarize_result(r_enh)
            s_gt  = _summarize_result(r_gt)

            row = {
                "image": stem,
                "low_n_det":      s_low["n_det"],
                "low_mean_conf":  s_low["mean_conf"],
                "low_n_classes":  s_low["n_classes"],
                "low_classes":    ";".join(s_low["class_names"]),
                "enh_n_det":      s_enh["n_det"],
                "enh_mean_conf":  s_enh["mean_conf"],
                "enh_n_classes":  s_enh["n_classes"],
                "enh_classes":    ";".join(s_enh["class_names"]),
                "gt_n_det":       s_gt["n_det"],
                "gt_mean_conf":   s_gt["mean_conf"],
                "gt_n_classes":   s_gt["n_classes"],
                "gt_classes":     ";".join(s_gt["class_names"]),
            }
            rows.append(row)

            # --- 누적 ---
            for key, summ in (("low", s_low), ("enh", s_enh), ("gt", s_gt)):
                sums[key]["n_det"]     += summ["n_det"]
                sums[key]["n_classes"] += summ["n_classes"]
                if summ["n_det"] > 0 and summ["mean_conf"] == summ["mean_conf"]:
                    # 검출이 있는 이미지만 confidence 평균에 포함
                    sums[key]["mean_conf"]   += summ["mean_conf"]
                    sums[key]["conf_weight"] += 1

            # --- 시각 비교 PNG ---
            low_annot = _annotated_rgb(r_low)
            enh_annot = _annotated_rgb(r_enh)
            gt_annot  = _annotated_rgb(r_gt)
            save_comparison_triplet(
                low_annot, enh_annot, gt_annot,
                out_path=results_dir / f"{stem}_compare.png",
            )

            # 진행 상황 한 줄 출력
            print(f"  [{stem:<10}] "
                  f"low={s_low['n_det']:>2}({_fmt_float(s_low['mean_conf'], 2)}) "
                  f"enh={s_enh['n_det']:>2}({_fmt_float(s_enh['mean_conf'], 2)}) "
                  f"gt={s_gt['n_det']:>2}({_fmt_float(s_gt['mean_conf'], 2)})")

    # ---- 전체 평균 ----
    n_pairs = max(len(rows), 1)
    overall: Dict[str, Dict[str, float]] = {}
    for key in ("low", "enh", "gt"):
        n_conf = sums[key]["conf_weight"]
        overall[key] = {
            "n_det":     sums[key]["n_det"]     / n_pairs,
            "n_classes": sums[key]["n_classes"] / n_pairs,
            "mean_conf": (sums[key]["mean_conf"] / n_conf) if n_conf > 0
                         else float("nan"),
        }

    print()
    print_comparison_table(rows, overall)

    # ---- CSV 저장 ----
    csv_path = results_dir / "detection_comparison.csv"
    fieldnames = [
        "image",
        "low_n_det", "low_mean_conf", "low_n_classes", "low_classes",
        "enh_n_det", "enh_mean_conf", "enh_n_classes", "enh_classes",
        "gt_n_det",  "gt_mean_conf",  "gt_n_classes",  "gt_classes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        # 평균 행 추가
        writer.writerow({
            "image": "AVG",
            "low_n_det":     f"{overall['low']['n_det']:.3f}",
            "low_mean_conf": _fmt_float(overall["low"]["mean_conf"], 4),
            "low_n_classes": f"{overall['low']['n_classes']:.3f}",
            "low_classes":   "",
            "enh_n_det":     f"{overall['enh']['n_det']:.3f}",
            "enh_mean_conf": _fmt_float(overall["enh"]["mean_conf"], 4),
            "enh_n_classes": f"{overall['enh']['n_classes']:.3f}",
            "enh_classes":   "",
            "gt_n_det":      f"{overall['gt']['n_det']:.3f}",
            "gt_mean_conf":  _fmt_float(overall["gt"]["mean_conf"], 4),
            "gt_n_classes":  f"{overall['gt']['n_classes']:.3f}",
            "gt_classes":    "",
        })
    print(f"  Saved CSV   → {csv_path}")
    print(f"  Saved PNGs  → {results_dir} ({len(rows)} files: *_compare.png)")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
