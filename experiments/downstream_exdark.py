"""ExDark Downstream 객체 검출 실험 — LUNA 향상 효과 정량 평가.

목적 (Purpose)
--------------
ExDark (Loh & Chan, CVIU 2019) 의 GT bounding-box annotation 을 기준으로,
* 원본 저조도 이미지   에 YOLOv8n(COCO pre-trained) 을 적용한 검출 성능
* LUNA(LightEnhanceGenerator) 로 향상한 이미지에 같은 YOLOv8n 을 적용한
  검출 성능
두 가지를 동일 GT 에 대해 **mAP@0.5 / mAP@0.5:0.95 / Precision / Recall**
로 비교한다. ``experiments/downstream_detection.py`` (LOL eval15 페어) 가
``"검출 수 / 평균 confidence"`` 수준의 *질적* 비교만 했다면, 본 스크립트는
GT bbox 가 있는 ExDark 로 *정량* 비교를 제공한다.

파이프라인
----------
1. ``DataSet/ExDark/annotations/imageclasslist.txt`` 에서 split=3(Test) 만 추출.
2. 각 테스트 이미지에 대해:
   * PIL 로 원본 로드 (해상도 보존)
   * LUNA 입력 = 256×256 BILINEAR 리사이즈 → tensor [-1,1] → ``G(x)``
   * LUNA 출력 = uint8 RGB → 원본 해상도로 다시 리사이즈 (BILINEAR)
3. 원본 / 향상 두 이미지에 YOLOv8n 추론 (원본 해상도 그대로).
4. ``annotations/<ClassName>/<imagename>.<ext>.txt`` 의 bbGt v3 파싱 → GT.
5. ExDark 12 클래스만을 평가 대상으로 두고 COCO 클래스 ID 로 매핑:
   ``Bicycle→1, Boat→8, Bottle→39, Bus→5, Car→2, Cat→15, Chair→56,
   Cup→41, Dog→16, Motorbike→3, People→0, Table→60``.
   YOLO 검출 결과 중 이 12 개 ID 만 평가에 포함.
6. mAP 계산기 (COCO-style):
   * per-class, per-IoU-threshold (0.50…0.95 step 0.05) Average Precision.
   * 모든 예측을 confidence 내림차순 정렬 → greedy GT 매칭 → PR 곡선 →
     all-point interpolation AP.
   * Precision / Recall 은 IoU=0.5 기준, 모든 (class, image) 검출을 합쳐서
     단일 값으로 보고.
7. 결과:
   * 콘솔 비교표 (per-class + overall mAP/P/R).
   * CSV: ``experiments/results/exdark/detection_comparison.csv``.
   * 시각 비교 PNG 10 장: ``experiments/results/exdark/visual/``.

사용 예
-------
.. code-block:: bash

    pip install ultralytics
    python experiments/downstream_exdark.py \\
        --checkpoint checkpoints/ext_lol_v2_real_stage2_best.pth

옵션
----
``--max_samples N`` : 테스트셋이 너무 크면 앞에서 N 개만 사용 (디버그).
``--all_splits``    : split 필터를 끄고 7363 장 전체로 평가 (시간 ↑↑).
``--num_visuals K`` : 시각 비교 PNG 장수 (기본 10).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- 프로젝트 루트를 sys.path 에 등록 (models/ 등 import) ---
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
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode

# 기존 LOL eval 스크립트의 helper 재사용 — generator 로드 / norm tensor 변환.
from experiments.downstream_detection import (  # type: ignore
    load_luna_generator,
    pil_to_norm_tensor,
    norm_tensor_to_uint8_rgb,
)


HRULE = "=" * 110
SUBRULE = "-" * 110


# ===========================================================================
# 1. ExDark 12 클래스 ↔ COCO 클래스 ID 매핑
# ===========================================================================
# YOLOv8 (ultralytics) 는 MS-COCO 80 클래스로 학습되어 있으므로 ExDark 의
# 라벨명을 COCO 클래스 ID 로 환산해야 동일 axis 에서 비교 가능하다.
EXDARK_TO_COCO: Dict[str, int] = {
    "Bicycle":   1,   # bicycle
    "Boat":      8,   # boat
    "Bottle":   39,   # bottle
    "Bus":       5,   # bus
    "Car":       2,   # car
    "Cat":      15,   # cat
    "Chair":    56,   # chair
    "Cup":      41,   # cup
    "Dog":      16,   # dog
    "Motorbike": 3,   # motorcycle
    "People":    0,   # person
    "Table":    60,   # dining table
}
EXDARK_CLASSES: Tuple[str, ...] = tuple(EXDARK_TO_COCO.keys())
TARGET_COCO_IDS: frozenset = frozenset(EXDARK_TO_COCO.values())
# 역방향 — 출력 라벨용 (COCO id → ExDark 이름)
COCO_TO_EXDARK: Dict[int, str] = {v: k for k, v in EXDARK_TO_COCO.items()}

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ===========================================================================
# 2. ExDark annotation (bbGt v3) + split 파싱
# ===========================================================================
def parse_bbgt_v3(ann_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """ExDark annotation 파일 1 개를 (coco_id, x1, y1, x2, y2) 리스트로 반환.

    파일 포맷::

        % bbGt version=3
        <ClassName> <l> <t> <w> <h> <occ1> ... <occ7>
        ...

    좌표는 픽셀 단위 (top-left + width/height) 이므로 (x1, y1, x2, y2) 로
    변환. ``ClassName`` 이 ExDark 12 클래스 외 (변형/오타) 면 무시.
    """
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not ann_path.is_file():
        return boxes
    try:
        text = ann_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return boxes
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_name = parts[0]
        # 일부 변형 (대소문자, 'people' 등) 대비
        canonical = None
        for c in EXDARK_CLASSES:
            if c.lower() == cls_name.lower():
                canonical = c
                break
        if canonical is None:
            continue
        try:
            l, t, w, h = (float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        coco_id = EXDARK_TO_COCO[canonical]
        boxes.append((coco_id, l, t, l + w, t + h))
    return boxes


def parse_imageclasslist(list_path: Path) -> Dict[str, int]:
    """``imageclasslist.txt`` → ``{image_filename: split_int}`` 매핑.

    공식 README 포맷::

        Name Class Light In/Out Train/Val/Test
        2015_00001.png 1 1 1 1
        ...

    헤더 줄이 있을 수도 있고 없을 수도 있어 둘 다 허용. split 값은 1/2/3
    이고 그 외 토큰이 들어오면 그 줄은 무시.
    """
    out: Dict[str, int] = {}
    if not list_path.is_file():
        return out
    for raw in list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = raw.strip().split()
        if len(toks) < 2:
            continue
        # 헤더 줄: 토큰이 모두 알파벳/슬래시 → 숫자 변환 실패하면 skip
        try:
            split = int(toks[-1])
        except ValueError:
            continue
        if split not in (1, 2, 3):
            continue
        out[toks[0]] = split
    return out


# ---------------------------------------------------------------------------
# ExDark 샘플 (이미지 1 장 + 그 이미지의 모든 GT bbox)
# ---------------------------------------------------------------------------
class ExDarkSample:
    """ExDark 테스트 샘플 1 개를 표현하는 경량 컨테이너.

    Attributes
    ----------
    image_path : Path
        ``images/<ClassName>/<filename>.<ext>`` 절대 경로.
    ann_path : Path
        ``annotations/<ClassName>/<filename>.<ext>.txt`` 절대 경로.
    class_dir : str
        해당 이미지가 속한 ExDark "메인" 클래스 폴더 이름. annotation 파일
        안에는 여러 클래스의 객체가 함께 들어있을 수 있으므로 이 값은 단지
        파일 경로 식별용 — 평가는 annotation 파싱 결과 전체를 사용.
    split : int
        1=Train, 2=Val, 3=Test.
    """

    __slots__ = ("image_path", "ann_path", "class_dir", "split")

    def __init__(self, image_path: Path, ann_path: Path,
                 class_dir: str, split: int) -> None:
        self.image_path = image_path
        self.ann_path = ann_path
        self.class_dir = class_dir
        self.split = split


def collect_exdark_samples(
    target_root: Path,
    splits: Optional[Tuple[int, ...]] = (3,),
) -> List[ExDarkSample]:
    """ExDark 폴더 트리에서 ``ExDarkSample`` 리스트를 만든다.

    Parameters
    ----------
    target_root : Path
        ``DataSet/ExDark`` (안에 ``images/``, ``annotations/`` 가 있음).
    splits : tuple[int, ...] | None
        포함할 split 번호. None 이면 전체.

    Returns
    -------
    list[ExDarkSample]
        파일명 사전순 정렬. 이미지 ↔ annotation 가 모두 존재하는 것만 포함.
    """
    images_root = target_root / "images"
    ann_root = target_root / "annotations"
    if not images_root.is_dir():
        raise FileNotFoundError(f"images 폴더 없음: {images_root}")
    if not ann_root.is_dir():
        raise FileNotFoundError(f"annotations 폴더 없음: {ann_root}")

    # split 사전 (없어도 동작은 가능 — 그 경우 splits=None 와 같음)
    split_map = parse_imageclasslist(ann_root / "imageclasslist.txt")
    if not split_map and splits is not None:
        print(f"  [warn] imageclasslist.txt 누락 또는 비어있음 — split 필터 무시")
        splits = None

    samples: List[ExDarkSample] = []
    for cls_name in EXDARK_CLASSES:
        img_dir = images_root / cls_name
        ann_dir = ann_root / cls_name
        if not img_dir.is_dir() or not ann_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
                continue
            ann_path = ann_dir / f"{img_path.name}.txt"
            if not ann_path.is_file():
                continue
            sp = split_map.get(img_path.name, 3)  # 없으면 Test 로 간주 (보수적)
            if splits is not None and sp not in splits:
                continue
            samples.append(ExDarkSample(
                image_path=img_path, ann_path=ann_path,
                class_dir=cls_name, split=sp,
            ))
    return samples


# ===========================================================================
# 3. LUNA 향상 — 원본 크기 복원
# ===========================================================================
def enhance_with_luna(
    G,
    pil_image: Image.Image,
    image_size: int,
    device: str,
) -> np.ndarray:
    """저조도 PIL → LUNA 향상 → uint8 RGB (원본 해상도) ndarray.

    LUNA 는 256×256 로 학습되었으므로 모델 입력은 BILINEAR 리사이즈로
    256×256 으로 맞추고, 출력을 다시 원본 (W, H) 로 복원한다 (BILINEAR).
    """
    W, H = pil_image.size
    norm = pil_to_norm_tensor(pil_image, image_size).to(device)
    enh_norm = G(norm)
    enh_rgb_256 = norm_tensor_to_uint8_rgb(enh_norm)  # (256, 256, 3) uint8
    if (H, W) != enh_rgb_256.shape[:2]:
        enh_pil = Image.fromarray(enh_rgb_256).resize(
            (W, H), resample=Image.BILINEAR,
        )
        return np.array(enh_pil, dtype=np.uint8)
    return enh_rgb_256


# ===========================================================================
# 4. YOLOv8 추론 결과 → (boxes_xyxy, classes, confs) 배열로 변환
# ===========================================================================
def _run_yolo(model, rgb: np.ndarray, conf: float, device: str):
    """RGB uint8 → ultralytics Results (BGR 입력 컨벤션 자동 변환)."""
    bgr = rgb[..., ::-1].copy()
    results = model.predict(bgr, conf=conf, verbose=False, device=device)
    return results[0]


def _extract_boxes(result) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ultralytics Results → (boxes_xyxy (N,4), classes (N,), confs (N,)).

    검출이 0 이면 (0, 4) / (0,) / (0,) 의 빈 배열을 반환.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32))
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
    # ExDark 12 클래스에 대응하는 COCO ID 만 유지
    mask = np.array([c in TARGET_COCO_IDS for c in cls], dtype=bool)
    return xyxy[mask], cls[mask], conf[mask]


# ===========================================================================
# 5. mAP / Precision / Recall 계산기
# ===========================================================================
def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU 행렬 — a (N,4), b (M,4) → (N, M).

    좌표는 (x1, y1, x2, y2). 빈 배열 입력 시 빈 행렬 반환.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    # 교집합
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _ap_all_point(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """PR 곡선 → all-point interpolation AP (Pascal VOC 2010 이후 + COCO).

    1) Precision 을 monotonic 감소로 보정.
    2) Recall 변화 구간마다 (Δr × p) 합산.

    빈 입력은 0.0 반환.
    """
    if precisions.size == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    # 모노톤 보정
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Δr 가 0 이 아닌 구간만
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class DetectionAccumulator:
    """이미지별 검출/GT 누적 → COCO-style mAP / P / R 계산.

    설계
    ----
    * 평가 대상 클래스는 ``TARGET_COCO_IDS`` (12 개) 로 고정.
    * 예측 / GT 를 누적해 두었다가 ``compute()`` 시 한 번에 계산.
    * IoU thresholds: 0.50, 0.55, …, 0.95 (10 개).
    * AP 는 all-point interpolation (Pascal VOC 2010+ 스타일, COCO 와 호환).
    * Precision / Recall 은 mAP 와 별개로, IoU=0.5 기준 전체 검출을 합쳐
      "TP/(TP+FP)" 와 "TP/(TP+FN)" 로 보고. (per-class P/R 도 함께.)
    """

    IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)  # 10 개

    def __init__(self) -> None:
        # 예측: list of (img_id, class_id, conf, x1, y1, x2, y2)
        self._preds: List[Tuple[int, int, float, float, float, float, float]] = []
        # GT: img_id → list of (class_id, x1, y1, x2, y2)
        self._gts: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
        self._img_ids: set = set()

    # ------------------------------------------------------------------
    def add(
        self,
        img_id: int,
        gt_boxes: np.ndarray, gt_classes: np.ndarray,
        pred_boxes: np.ndarray, pred_classes: np.ndarray, pred_confs: np.ndarray,
    ) -> None:
        """이미지 1 장의 GT / 예측 추가. 좌표는 모두 (x1,y1,x2,y2) 원본 픽셀."""
        self._img_ids.add(img_id)
        for (x1, y1, x2, y2), c in zip(gt_boxes, gt_classes):
            if int(c) in TARGET_COCO_IDS:
                self._gts.setdefault(img_id, []).append(
                    (int(c), float(x1), float(y1), float(x2), float(y2))
                )
        for (x1, y1, x2, y2), c, conf in zip(pred_boxes, pred_classes, pred_confs):
            if int(c) in TARGET_COCO_IDS:
                self._preds.append(
                    (img_id, int(c), float(conf),
                     float(x1), float(y1), float(x2), float(y2))
                )

    # ------------------------------------------------------------------
    def _ap_for_class(
        self, cls_id: int, iou_th: float,
    ) -> Tuple[float, int, int, int]:
        """단일 (class, IoU) 의 AP + (TP_total, FP_total, n_gt) 반환.

        TP/FP 누적은 IoU=0.5 의 P/R 계산에도 재사용.
        """
        # 클래스 필터
        preds = [p for p in self._preds if p[1] == cls_id]
        preds.sort(key=lambda x: -x[2])  # confidence 내림차순

        # GT: img_id → 남은 인덱스 리스트 (greedy 매칭 시 소비)
        gts_per_img: Dict[int, List[Tuple[float, float, float, float]]] = {}
        for img_id, boxes in self._gts.items():
            xs = [(x1, y1, x2, y2) for (c, x1, y1, x2, y2) in boxes if c == cls_id]
            if xs:
                gts_per_img[img_id] = xs
        n_gt = sum(len(v) for v in gts_per_img.values())
        if n_gt == 0 or not preds:
            return 0.0, 0, len(preds), n_gt

        matched: Dict[int, set] = {k: set() for k in gts_per_img}
        tps = np.zeros(len(preds), dtype=np.float32)
        fps = np.zeros(len(preds), dtype=np.float32)

        for i, (img_id, _c, _conf, x1, y1, x2, y2) in enumerate(preds):
            gts = gts_per_img.get(img_id)
            if not gts:
                fps[i] = 1.0
                continue
            # 이 prediction 과 가장 IoU 높은 unmatched GT 찾기
            box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            gt_arr = np.array(gts, dtype=np.float32)
            ious = _box_iou_xyxy(box, gt_arr)[0]
            # 이미 매칭된 GT 는 -1 로 마스킹
            for j in matched[img_id]:
                ious[j] = -1.0
            best_j = int(np.argmax(ious))
            best_iou = float(ious[best_j])
            if best_iou >= iou_th:
                tps[i] = 1.0
                matched[img_id].add(best_j)
            else:
                fps[i] = 1.0

        # PR 곡선
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = _ap_all_point(precision, recall)
        return ap, int(tps.sum()), int(fps.sum()), n_gt

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, Any]:
        """모든 누적 결과 → 전체/클래스별 mAP, P, R 보고.

        Returns
        -------
        dict
            ``{
                "map50":  float,      # mAP @ IoU=0.5
                "map":    float,      # mAP @ IoU=0.5:0.95
                "p50":    float,      # Precision (IoU=0.5, 전체 클래스 합산)
                "r50":    float,      # Recall    (IoU=0.5, 전체 클래스 합산)
                "per_class": {
                    coco_id (int): {"name": str, "n_gt": int, "n_pred": int,
                                    "ap50": float, "ap": float,
                                    "p": float, "r": float}
                }
            }``
        """
        per_class: Dict[int, Dict[str, Any]] = {}
        ap50_list: List[float] = []
        ap_list:   List[float] = []
        total_tp = 0
        total_fp = 0
        total_gt = 0

        for cls_id in sorted(TARGET_COCO_IDS):
            ap_per_iou: List[float] = []
            tp_at_50, fp_at_50, n_gt = 0, 0, 0
            for k, iou_th in enumerate(self.IOU_THRESHOLDS):
                ap, tp_c, fp_c, gt_c = self._ap_for_class(cls_id, float(iou_th))
                ap_per_iou.append(ap)
                if k == 0:  # IoU = 0.5
                    tp_at_50, fp_at_50, n_gt = tp_c, fp_c, gt_c
            if n_gt == 0:
                # 평가 대상 0 → 통계 제외
                continue
            ap50 = ap_per_iou[0]
            ap = float(np.mean(ap_per_iou))
            p_c = tp_at_50 / max(tp_at_50 + fp_at_50, 1)
            r_c = tp_at_50 / max(n_gt, 1)
            per_class[cls_id] = {
                "name":   COCO_TO_EXDARK.get(cls_id, str(cls_id)),
                "n_gt":   int(n_gt),
                "n_pred": int(tp_at_50 + fp_at_50),
                "ap50":   float(ap50),
                "ap":     float(ap),
                "p":      float(p_c),
                "r":      float(r_c),
            }
            ap50_list.append(ap50)
            ap_list.append(ap)
            total_tp += tp_at_50
            total_fp += fp_at_50
            total_gt += n_gt

        return {
            "map50":     float(np.mean(ap50_list)) if ap50_list else 0.0,
            "map":       float(np.mean(ap_list))   if ap_list   else 0.0,
            "p50":       total_tp / max(total_tp + total_fp, 1),
            "r50":       total_tp / max(total_gt, 1),
            "per_class": per_class,
            "n_images":  len(self._img_ids),
            "n_preds":   len(self._preds),
            "n_gts":     sum(len(v) for v in self._gts.values()),
        }


# ===========================================================================
# 6. 시각화 — [Original+pred | Enhanced+pred | Original+GT] 가로 배치
# ===========================================================================
# 색상 (RGB) — bbox 종류별로 구분
_COLOR_PRED = (255, 64, 64)    # 빨강: YOLO 예측 (원본)
_COLOR_PRED_ENH = (64, 200, 64)  # 초록: YOLO 예측 (향상)
_COLOR_GT = (64, 160, 255)     # 파랑: GT


def _draw_boxes(
    rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: List[str],
    color: Tuple[int, int, int],
    width: int = 2,
) -> np.ndarray:
    """RGB uint8 에 bbox + 라벨 텍스트를 그려 반환."""
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    for (x1, y1, x2, y2), lab in zip(boxes_xyxy, labels):
        draw.rectangle([float(x1), float(y1), float(x2), float(y2)],
                       outline=color, width=width)
        # 라벨 배경 (가독성)
        try:
            l, t, r, b = draw.textbbox((0, 0), lab)
            tw, th = r - l, b - t
        except Exception:
            tw, th = 8 * len(lab), 10
        ty = max(int(y1) - th - 2, 0)
        draw.rectangle([float(x1), ty, float(x1) + tw + 4, ty + th + 2],
                       fill=color)
        draw.text((float(x1) + 2, ty + 1), lab, fill=(255, 255, 255))
    return np.array(pil, dtype=np.uint8)


def save_visual_comparison(
    orig_rgb: np.ndarray,
    enh_rgb: np.ndarray,
    pred_orig: Tuple[np.ndarray, np.ndarray, np.ndarray],
    pred_enh:  Tuple[np.ndarray, np.ndarray, np.ndarray],
    gt_boxes:  np.ndarray, gt_classes: np.ndarray,
    out_path: Path,
    pad: int = 6,
) -> None:
    """단일 이미지에 대한 3-panel 시각 비교 PNG 저장.

    좌:  원본 + YOLO 예측 (red)
    중:  LUNA 향상 + YOLO 예측 (green)
    우:  원본 + GT (blue)
    """
    def _labels(classes: np.ndarray, confs: Optional[np.ndarray]) -> List[str]:
        out = []
        for i, c in enumerate(classes):
            name = COCO_TO_EXDARK.get(int(c), str(int(c)))
            if confs is None:
                out.append(name)
            else:
                out.append(f"{name} {confs[i]:.2f}")
        return out

    po_boxes, po_cls, po_conf = pred_orig
    pe_boxes, pe_cls, pe_conf = pred_enh

    panel_orig = _draw_boxes(orig_rgb, po_boxes, _labels(po_cls, po_conf), _COLOR_PRED)
    panel_enh  = _draw_boxes(enh_rgb,  pe_boxes, _labels(pe_cls, pe_conf), _COLOR_PRED_ENH)
    panel_gt   = _draw_boxes(orig_rgb, gt_boxes, _labels(gt_classes, None), _COLOR_GT)

    h, w, _ = panel_orig.shape
    sep = np.full((h, pad, 3), 255, dtype=np.uint8)
    concat = np.concatenate([panel_orig, sep, panel_enh, sep, panel_gt], axis=1)

    # 상단 라벨 띠
    label_h = 24
    bar = np.full((label_h, concat.shape[1], 3), 245, dtype=np.uint8)
    img = np.concatenate([bar, concat], axis=0)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    centers_x = [w // 2, w + pad + w // 2, 2 * (w + pad) + w // 2]
    titles = (
        f"Original ({len(po_boxes)} det)",
        f"Enhanced ({len(pe_boxes)} det)",
        f"GT ({len(gt_boxes)} obj)",
    )
    for cx, label in zip(centers_x, titles):
        try:
            l, t, r, b = draw.textbbox((0, 0), label)
            tw = r - l
        except Exception:
            tw = 8 * len(label)
        draw.text((cx - tw // 2, 5), label, fill=(0, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


# ===========================================================================
# 7. 콘솔 비교표
# ===========================================================================
def _fmt(x: float, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{d}f}"


def print_comparison_table(
    res_orig: Dict[str, Any],
    res_enh: Dict[str, Any],
) -> None:
    """per-class + overall 비교표를 stdout 에 출력 (Original vs Enhanced)."""
    print(HRULE)
    print(" ExDark Downstream Detection — Original (low) vs Enhanced (LUNA)  /  GT = ExDark annotation")
    print(HRULE)
    print(f"  {'Class':<14} | {'#GT':>5} | "
          f"{'Orig AP50':>10} {'Enh AP50':>9} {'ΔAP50':>8} | "
          f"{'Orig AP':>9} {'Enh AP':>8} {'ΔAP':>8} | "
          f"{'Orig P':>7} {'Enh P':>7} | {'Orig R':>7} {'Enh R':>7}")
    print(SUBRULE)
    classes = sorted(set(res_orig["per_class"].keys()) | set(res_enh["per_class"].keys()))
    for cid in classes:
        po = res_orig["per_class"].get(cid)
        pe = res_enh["per_class"].get(cid)
        name = (po or pe)["name"]
        n_gt = (po or pe)["n_gt"]
        ap50_o = po["ap50"] if po else 0.0
        ap50_e = pe["ap50"] if pe else 0.0
        ap_o   = po["ap"]   if po else 0.0
        ap_e   = pe["ap"]   if pe else 0.0
        p_o    = po["p"]    if po else 0.0
        p_e    = pe["p"]    if pe else 0.0
        r_o    = po["r"]    if po else 0.0
        r_e    = pe["r"]    if pe else 0.0
        d50 = ap50_e - ap50_o
        d   = ap_e   - ap_o
        print(f"  {name:<14} | {n_gt:>5} | "
              f"{_fmt(ap50_o, 3):>10} {_fmt(ap50_e, 3):>9} {d50:+.3f} | "
              f"{_fmt(ap_o, 3):>9} {_fmt(ap_e, 3):>8} {d:+.3f} | "
              f"{_fmt(p_o, 3):>7} {_fmt(p_e, 3):>7} | "
              f"{_fmt(r_o, 3):>7} {_fmt(r_e, 3):>7}")
    print(SUBRULE)
    print(f"  {'OVERALL':<14} | {res_orig['n_gts']:>5} | "
          f"{_fmt(res_orig['map50'], 3):>10} {_fmt(res_enh['map50'], 3):>9} "
          f"{(res_enh['map50']-res_orig['map50']):+.3f} | "
          f"{_fmt(res_orig['map'], 3):>9} {_fmt(res_enh['map'], 3):>8} "
          f"{(res_enh['map']-res_orig['map']):+.3f} | "
          f"{_fmt(res_orig['p50'], 3):>7} {_fmt(res_enh['p50'], 3):>7} | "
          f"{_fmt(res_orig['r50'], 3):>7} {_fmt(res_enh['r50'], 3):>7}")
    print(HRULE)
    # 한 줄 요약 (논문 본문용)
    print(f"  ΔmAP@0.5      : {(res_enh['map50']-res_orig['map50']):+.4f} "
          f"({_fmt(res_orig['map50'], 4)} → {_fmt(res_enh['map50'], 4)})")
    print(f"  ΔmAP@0.5:0.95 : {(res_enh['map']-res_orig['map']):+.4f} "
          f"({_fmt(res_orig['map'], 4)} → {_fmt(res_enh['map'], 4)})")
    print(f"  ΔPrecision    : {(res_enh['p50']-res_orig['p50']):+.4f}")
    print(f"  ΔRecall       : {(res_enh['r50']-res_orig['r50']):+.4f}")
    print(HRULE)


# ===========================================================================
# 8. CSV 저장
# ===========================================================================
def save_results_csv(
    res_orig: Dict[str, Any],
    res_enh: Dict[str, Any],
    out_path: Path,
) -> None:
    """클래스별 + OVERALL 행을 가진 CSV 저장."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class", "n_gt",
        "orig_n_pred", "enh_n_pred",
        "orig_ap50", "enh_ap50", "delta_ap50",
        "orig_ap",   "enh_ap",   "delta_ap",
        "orig_p",    "enh_p",    "delta_p",
        "orig_r",    "enh_r",    "delta_r",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        classes = sorted(set(res_orig["per_class"].keys())
                         | set(res_enh["per_class"].keys()))
        for cid in classes:
            po = res_orig["per_class"].get(cid)
            pe = res_enh["per_class"].get(cid)
            name = (po or pe)["name"]
            n_gt = (po or pe)["n_gt"]

            def g(d: Optional[Dict[str, Any]], key: str) -> float:
                return float(d[key]) if d else 0.0

            row = {
                "class": name, "n_gt": n_gt,
                "orig_n_pred": po["n_pred"] if po else 0,
                "enh_n_pred":  pe["n_pred"] if pe else 0,
                "orig_ap50":   f"{g(po,'ap50'):.6f}",
                "enh_ap50":    f"{g(pe,'ap50'):.6f}",
                "delta_ap50":  f"{g(pe,'ap50')-g(po,'ap50'):+.6f}",
                "orig_ap":     f"{g(po,'ap'):.6f}",
                "enh_ap":      f"{g(pe,'ap'):.6f}",
                "delta_ap":    f"{g(pe,'ap')-g(po,'ap'):+.6f}",
                "orig_p":      f"{g(po,'p'):.6f}",
                "enh_p":       f"{g(pe,'p'):.6f}",
                "delta_p":     f"{g(pe,'p')-g(po,'p'):+.6f}",
                "orig_r":      f"{g(po,'r'):.6f}",
                "enh_r":       f"{g(pe,'r'):.6f}",
                "delta_r":     f"{g(pe,'r')-g(po,'r'):+.6f}",
            }
            w.writerow(row)
        # OVERALL
        w.writerow({
            "class": "OVERALL", "n_gt": res_orig["n_gts"],
            "orig_n_pred": res_orig["n_preds"], "enh_n_pred": res_enh["n_preds"],
            "orig_ap50":   f"{res_orig['map50']:.6f}",
            "enh_ap50":    f"{res_enh['map50']:.6f}",
            "delta_ap50":  f"{res_enh['map50']-res_orig['map50']:+.6f}",
            "orig_ap":     f"{res_orig['map']:.6f}",
            "enh_ap":      f"{res_enh['map']:.6f}",
            "delta_ap":    f"{res_enh['map']-res_orig['map']:+.6f}",
            "orig_p":      f"{res_orig['p50']:.6f}",
            "enh_p":       f"{res_enh['p50']:.6f}",
            "delta_p":     f"{res_enh['p50']-res_orig['p50']:+.6f}",
            "orig_r":      f"{res_orig['r50']:.6f}",
            "enh_r":       f"{res_enh['r50']:.6f}",
            "delta_r":     f"{res_enh['r50']-res_orig['r50']:+.6f}",
        })


# ===========================================================================
# 9. Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ExDark Downstream 검출 평가 — 원본 vs LUNA 향상 (mAP/P/R)",
    )
    p.add_argument(
        "--exdark_root", type=str,
        default=r"C:\대학교\Projects\SmallSizePM_GAN_model\DataSet\ExDark",
        help="ExDark 데이터셋 루트 (images/, annotations/ 포함)",
    )
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/ext_lol_v2_real_stage2_best.pth",
                   help="LUNA generator 가중치 (hybrid_v1 호환)")
    p.add_argument("--yolo_weights", type=str, default="yolov8n.pt",
                   help="YOLOv8 가중치 (없으면 ultralytics 가 자동 다운로드)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="YOLO confidence threshold (검출 기준)")
    p.add_argument("--image_size", type=int, default=256,
                   help="LUNA 입력 해상도 (학습과 동일하게 유지)")
    p.add_argument("--results_dir", type=str,
                   default="experiments/results/exdark",
                   help="CSV / 시각 비교 저장 디렉토리")
    p.add_argument("--num_visuals", type=int, default=10,
                   help="시각 비교 PNG 장수 (GT bbox 가 많은 이미지 우선)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 이면 split 필터 결과 전체, 양수면 앞에서 N 개만 사용")
    p.add_argument("--all_splits", action="store_true",
                   help="split 필터 끄고 7363 장 전체 사용 (Train+Val+Test).")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (기본: cuda 가용 시 cuda)")
    p.add_argument("--seed", type=int, default=0,
                   help="시각 비교 샘플 선정 시 tie-break 재현성")
    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _try_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except ImportError:
        return None


def main() -> int:
    args = parse_args()
    device = args.device
    random.seed(args.seed)

    # --- ultralytics import (없으면 안내) ---
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics 미설치.  pip install ultralytics")
        return 1

    exdark_root = Path(args.exdark_root)
    ckpt_path = Path(args.checkpoint)
    results_dir = Path(args.results_dir).resolve()
    visual_dir = results_dir / "visual"
    results_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    print(HRULE)
    print(" ExDark Downstream Detection (YOLOv8n) — Original vs LUNA-Enhanced")
    print(HRULE)
    print(f"  exdark_root  : {exdark_root}")
    print(f"  checkpoint   : {ckpt_path}")
    print(f"  yolo_weights : {args.yolo_weights}")
    print(f"  conf_thresh  : {args.conf}")
    print(f"  image_size   : {args.image_size}  (LUNA 입력)")
    print(f"  device       : {device}")
    print(f"  results_dir  : {results_dir}")
    print(SUBRULE)

    # --- 체크포인트 / 데이터셋 존재 확인 ---
    if not ckpt_path.is_file():
        print(f"[error] LUNA 체크포인트가 없습니다: {ckpt_path}")
        return 1
    if not (exdark_root / "images").is_dir() or not (exdark_root / "annotations").is_dir():
        print(f"[error] ExDark 폴더 구조 누락:")
        print(f"        {exdark_root / 'images'}      exists={(exdark_root / 'images').is_dir()}")
        print(f"        {exdark_root / 'annotations'} exists={(exdark_root / 'annotations').is_dir()}")
        print(f"        먼저 다음을 실행하세요:")
        print(f"          python datasets/download_exdark.py")
        return 1

    # --- LUNA generator ---
    G = load_luna_generator(ckpt_path, device=device)
    n_params = sum(p.numel() for p in G.parameters())
    print(f"  LUNA params  : {n_params:,}  ({n_params/1e3:.1f} K)")

    # --- YOLOv8 ---
    print(f"  YOLO loading : {args.yolo_weights} ...")
    yolo = YOLO(args.yolo_weights)
    print(f"  YOLO classes : {len(yolo.names)} (COCO pre-trained)")
    print(SUBRULE)

    # --- ExDark 샘플 수집 ---
    splits = None if args.all_splits else (3,)
    samples = collect_exdark_samples(exdark_root, splits=splits)
    if not samples:
        print(f"[error] 샘플이 0 개입니다. imageclasslist.txt / annotation 파일 확인 필요.")
        return 1
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    split_desc = "Test only (split=3)" if not args.all_splits else "ALL splits (1+2+3)"
    print(f"  samples      : {len(samples)}  [{split_desc}]")
    print(SUBRULE)

    acc_orig = DetectionAccumulator()
    acc_enh  = DetectionAccumulator()

    # 시각화 후보 — (img_id, n_gt, sample) 를 모아두었다가 GT 많은 순으로 K 개
    visual_candidates: List[Tuple[int, int, ExDarkSample, np.ndarray, np.ndarray,
                                  Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  np.ndarray, np.ndarray]] = []

    tqdm = _try_tqdm()
    iterator = tqdm(samples, desc="ExDark", unit="img", ncols=100) if tqdm else samples

    with torch.no_grad():
        for img_id, sm in enumerate(iterator):
            try:
                pil = Image.open(sm.image_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] 이미지 로드 실패 {sm.image_path.name}: {e}")
                continue
            W, H = pil.size
            orig_rgb = np.array(pil, dtype=np.uint8)

            # GT 파싱 (원본 픽셀 좌표)
            gt_records = parse_bbgt_v3(sm.ann_path)
            if gt_records:
                gt_boxes = np.array([[x1, y1, x2, y2] for (_c, x1, y1, x2, y2) in gt_records],
                                    dtype=np.float32)
                gt_classes = np.array([c for (c, *_r) in gt_records], dtype=np.int64)
            else:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
                gt_classes = np.zeros((0,), dtype=np.int64)

            # LUNA 향상 (원본 크기 복원)
            enh_rgb = enhance_with_luna(G, pil, args.image_size, device)

            # YOLO 추론 (원본 / 향상)
            r_orig = _run_yolo(yolo, orig_rgb, args.conf, device)
            r_enh  = _run_yolo(yolo, enh_rgb,  args.conf, device)
            pb_orig, pc_orig, pf_orig = _extract_boxes(r_orig)
            pb_enh,  pc_enh,  pf_enh  = _extract_boxes(r_enh)

            # 누적
            acc_orig.add(img_id, gt_boxes, gt_classes, pb_orig, pc_orig, pf_orig)
            acc_enh.add(img_id,  gt_boxes, gt_classes, pb_enh,  pc_enh,  pf_enh)

            # 시각화 후보 보관 (GT 가 1 개 이상)
            if len(gt_boxes) > 0:
                visual_candidates.append((
                    img_id, int(len(gt_boxes)), sm,
                    orig_rgb, enh_rgb,
                    (pb_orig, pc_orig, pf_orig),
                    (pb_enh,  pc_enh,  pf_enh),
                    gt_boxes, gt_classes,
                ))

    # ---- 평가 ----
    print()
    print(" 평가 중 (mAP / Precision / Recall) ...")
    res_orig = acc_orig.compute()
    res_enh  = acc_enh.compute()

    print()
    print_comparison_table(res_orig, res_enh)

    # ---- CSV ----
    csv_path = results_dir / "detection_comparison.csv"
    save_results_csv(res_orig, res_enh, csv_path)
    print(f"  Saved CSV   → {csv_path}")

    # ---- 시각 비교 (GT 많은 순으로 num_visuals 개) ----
    k = max(min(args.num_visuals, len(visual_candidates)), 0)
    if k > 0:
        visual_candidates.sort(key=lambda t: (-t[1], t[0]))
        chosen = visual_candidates[:k]
        for (img_id, _ng, sm, orig_rgb, enh_rgb, pred_o, pred_e, gb, gc) in chosen:
            out = visual_dir / f"{sm.class_dir}_{sm.image_path.stem}_compare.png"
            try:
                save_visual_comparison(
                    orig_rgb=orig_rgb, enh_rgb=enh_rgb,
                    pred_orig=pred_o, pred_enh=pred_e,
                    gt_boxes=gb, gt_classes=gc,
                    out_path=out,
                )
            except Exception as e:
                print(f"  [warn] 시각화 실패 {sm.image_path.name}: {e}")
        print(f"  Saved PNGs  → {visual_dir} ({k} files)")
    else:
        print(f"  [info] 시각 비교 PNG 생성 0 — GT 가 있는 샘플이 없습니다.")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
