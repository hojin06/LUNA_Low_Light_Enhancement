"""ExDark detection-training Dataset + YOLO-format collate.

용도 (Purpose)
--------------
``train_detection_loss.py`` 가 ExDark 의 *어두운 이미지 + GT bounding-box
annotation* 을 YOLOv8 detection loss (``v8DetectionLoss``) 의 입력으로 직접
공급하기 위한 데이터셋.

기존 ``experiments/downstream_exdark.py`` 는 evaluation 전용으로 PIL 단위로
이미지를 순회하지만, 여기서는 ``torch.utils.data.Dataset`` 로 감싸 학습용
DataLoader 에 넣고 multi-worker / shuffle / batch 를 활용한다.

ExDark → COCO 클래스 매핑
-------------------------
YOLOv8n 은 COCO 80 classes 로 pretrained 이므로, ExDark 12 classes 의
annotation 을 그대로 사용하면 class id 가 어긋난다.  ``parse_bbgt_v3`` 가
이미 ``EXDARK_TO_COCO`` 로 환산된 ``coco_id`` 를 반환하므로, 본 데이터셋은
그 결과를 그대로 사용한다.  v8DetectionLoss 에 들어가는 ``cls`` 텐서는 COCO
ID (0~79 중 12 개 sparse subset).

좌표계 변환
-----------
* annotation 의 bbox 는 원본 픽셀 (x1, y1, x2, y2).
* 이미지 → 256×256 BILINEAR resize.
* bbox 는 **원본 (W, H) 로 정규화 후 (cx, cy, w, h)** 로 변환.  resize 는
  비율-uniform 이 아닐 수 있지만 normalized 좌표는 invariant.
* w 또는 h ≤ 0 인 잘못된 box 는 제거. clamp [0, 1].
* 최종 target row: ``[coco_class_id, cx, cy, w, h]`` (5-tuple).

Empty-targets 처리
------------------
GT bbox 가 한 개도 없거나 모두 제거된 sample 은 학습 신호가 없으므로
``__init__`` 단계에서 미리 걸러낸다 (filter_empty=True 가 기본).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# 기존 ExDark 인프라 재사용 (downstream_exdark 가 sys.path 를 알아서 잡아줌)
from experiments.downstream_exdark import (  # type: ignore
    EXDARK_TO_COCO,
    ExDarkSample,
    collect_exdark_samples,
    parse_bbgt_v3,
)


class ExDarkDetectionDataset(Dataset):
    """ExDark 이미지 + GT bbox (YOLO 포맷 targets) 를 반환하는 Dataset.

    Parameters
    ----------
    exdark_root : Path
        ``DataSet/ExDark`` 루트 (``images/``, ``annotations/`` 포함).
    image_size : int
        이미지 resize 후 H = W.  기본 256.
    splits : tuple[int, ...] | None
        ``(1, 2, 3)`` 이면 전체, ``(1, 2)`` 면 Train+Val, ``(3,)`` 면 Test.
        None 이면 split 필터 끔.
    max_samples : int
        0 = 무제한, 양수 = 앞에서 N 개만 사용 (디버그용).
    filter_empty : bool
        GT bbox 가 0 개인 sample 제거 (default True).
        v8DetectionLoss 가 empty target 도 처리 가능하지만 학습 신호가 없어
        시간 낭비이므로 기본 제외.
    """

    def __init__(
        self,
        exdark_root: Path,
        image_size: int = 256,
        splits: Optional[Tuple[int, ...]] = None,
        max_samples: int = 0,
        filter_empty: bool = True,
    ) -> None:
        super().__init__()
        self.exdark_root = Path(exdark_root)
        self.image_size = int(image_size)

        raw = collect_exdark_samples(self.exdark_root, splits=splits)
        # ExDark 의 annotation 을 한 번씩 미리 파싱하여 비어있는 sample 제거.
        # (학습 step 마다 IO 가 도는 게 아니라 init 1 회로 끝나므로 비용 부담 미미.)
        self.samples: List[ExDarkSample] = []
        self.cached_targets: List[List[Tuple[int, float, float, float, float]]] = []
        for sm in raw:
            recs = parse_bbgt_v3(sm.ann_path)
            if filter_empty and not recs:
                continue
            self.samples.append(sm)
            self.cached_targets.append(recs)

        if max_samples > 0:
            self.samples = self.samples[:max_samples]
            self.cached_targets = self.cached_targets[:max_samples]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sm = self.samples[idx]
        recs = self.cached_targets[idx]
        pil = Image.open(sm.image_path).convert("RGB")
        W, H = pil.size  # 원본 픽셀 크기 (정규화 분모)

        # ---- Image: 256 BILINEAR resize → tensor [-1, 1] ----
        pil_r = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = TF.to_tensor(pil_r) * 2.0 - 1.0  # (3, H, W) in [-1, 1]

        # ---- Targets: (cls, cx, cy, w, h)  in normalized [0, 1] ----
        target_rows: List[List[float]] = []
        for (cid, x1, y1, x2, y2) in recs:
            cx = (x1 + x2) * 0.5 / W
            cy = (y1 + y2) * 0.5 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            # boundary clamp + sanity
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            if bw <= 0.0 or bh <= 0.0:
                continue
            target_rows.append([float(cid), cx, cy, bw, bh])

        if target_rows:
            t = torch.tensor(target_rows, dtype=torch.float32)  # (N, 5)
        else:
            # filter_empty=True 면 이 branch 는 거의 안 탐.  방어 코드.
            t = torch.zeros((0, 5), dtype=torch.float32)
        return img, t


# ===========================================================================
# Collate — per-image targets 를 v8DetectionLoss 가 받는 batch dict 으로 합침
# ===========================================================================
def exdark_yolo_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """``(image, targets)`` list → ``(images_stacked, target_dict)``.

    target_dict 포맷 (v8DetectionLoss 가 기대):
        ``{
            "cls":       Tensor (N, 1),
            "bboxes":    Tensor (N, 4)  in normalized cx/cy/w/h,
            "batch_idx": Tensor (N,)    이미지 index 0..B-1,
        }``

    빈 배치 (모든 sample 의 target 이 0 개) 의 경우에도 동작.
    """
    imgs, targets = zip(*batch)
    imgs_stacked = torch.stack(imgs, dim=0)  # (B, 3, H, W)

    cls_parts:   List[torch.Tensor] = []
    bbox_parts:  List[torch.Tensor] = []
    bi_parts:    List[torch.Tensor] = []
    for i, t in enumerate(targets):
        if t.size(0) == 0:
            continue
        cls_parts.append(t[:, 0:1])                   # (Ni, 1)
        bbox_parts.append(t[:, 1:5])                  # (Ni, 4)
        bi_parts.append(torch.full((t.size(0),), i, dtype=torch.float32))

    if cls_parts:
        cls    = torch.cat(cls_parts,  dim=0)
        bboxes = torch.cat(bbox_parts, dim=0)
        bidx   = torch.cat(bi_parts,   dim=0)
    else:
        cls    = torch.zeros((0, 1), dtype=torch.float32)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
        bidx   = torch.zeros((0,),   dtype=torch.float32)

    return imgs_stacked, {"cls": cls, "bboxes": bboxes, "batch_idx": bidx}


# ===========================================================================
# Convenience helper — Loss 입력 직전 device 이동
# ===========================================================================
def move_target_dict(d: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """``target_dict`` 의 모든 텐서를 ``device`` 로 옮긴 새 dict 반환."""
    return {k: v.to(device, non_blocking=True) for k, v in d.items()}


# 디버그용: 사용 가능 클래스 명세를 노출
__all__ = [
    "ExDarkDetectionDataset",
    "exdark_yolo_collate",
    "move_target_dict",
    "EXDARK_TO_COCO",
]
