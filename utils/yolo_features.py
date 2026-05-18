"""Frozen YOLOv8n multi-level feature extractor (P3/P4/P5).

설계 목적 (Why)
---------------
LUNA 가 단순히 사람 눈에 좋은 이미지가 아니라, 다운스트림 검출기 YOLOv8n
이 "GT 와 동일한 detection feature" 를 추출하도록 향상시키기 위한
**reference feature extractor** 다.  YOLOv8n 자체는 절대 업데이트하지 않으며
LUNA 의 학습 신호만 통과시키는 frozen teacher 역할.

주의 (Gradient flow)
--------------------
* YOLO 의 모든 ``Parameter.requires_grad = False`` 로 동결.
* 하지만 forward 는 ``torch.no_grad`` 로 감싸지 않는다 — LUNA output 에서
  들어오는 gradient 가 YOLO 의 Conv/BN 을 *통과* 해서 LUNA 까지 역전파될 수
  있어야 하기 때문.  YOLO 가중치 자체는 grad 가 없으므로 update 되지 않는다.
* GT/Low branch 의 feature 는 *target* 이므로 caller 가 ``.detach()`` 또는
  ``torch.no_grad`` 로 분리해도 OK (DetectionAwareLoss 가 그렇게 한다).

P3/P4/P5 자동 탐지
------------------
YOLOv8n 구조는 ultralytics 버전마다 layer 번호가 달라질 수 있으므로
하드코딩하지 않는다.  대신 dummy forward 로 모든 layer 출력의 spatial size 를
기록하고, ``input_size`` 기준으로

    P3 → input_size / 8   (256 → 32)
    P4 → input_size / 16  (256 → 16)
    P5 → input_size / 32  (256 → 8)

에 해당하는 **마지막 layer** 를 선택한다.  YOLOv8n 의 경우 보통 neck 의
C2f 출력 (예: layer 15 / 18 / 21) 이 잡힌다 (= 검출 head 직전의 multi-scale
fused feature).

Skip-connection 처리
--------------------
ultralytics 의 module 은 ``m.f`` 속성으로 입력 source 를 지정한다.
``f == -1`` 이면 직전 출력, 정수면 saved 출력 index, 리스트면 여러 source
(Concat) 를 의미한다.  본 클래스는 ``forward`` 중 모든 중간 출력을 list 에
저장한 뒤 ``m.f`` 에 따라 골라 입력으로 넣는 ultralytics 내부 forward 와
동일한 방식으로 동작.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ===========================================================================
# YOLOv8n multi-level feature extractor (frozen, backbone+neck)
# ===========================================================================
class YOLOFeatureExtractor(nn.Module):
    """Frozen YOLOv8n backbone+neck — [0,1] RGB → {P3, P4, P5} feature dict.

    Parameters
    ----------
    weights : str
        ``yolov8n.pt`` 경로 (없으면 ultralytics 가 자동 다운로드).
    input_size : int
        P3/P4/P5 spatial size 추정을 위한 reference 해상도. LUNA 학습 입력과
        동일하게 256 권장.
    debug : bool
        True 면 ``__init__`` 끝에 전체 layer 목록 (i / type / output shape) 을
        stdout 에 출력 → P3/P4/P5 선택 근거 확인용.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        input_size: int = 256,
        debug: bool = False,
    ) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics 패키지가 필요합니다.  pip install ultralytics"
            ) from e

        yolo = YOLO(weights)
        # DetectionModel 의 내부 ModuleList — 0~N-1 layer + 마지막은 Detect head
        full = yolo.model.model

        # Detect head 는 학습 신호용이 아니므로 제외 (직전 layer 까지 보존)
        layers: List[nn.Module] = []
        for m in full:
            if type(m).__name__ == "Detect":
                break
            layers.append(m)
        if not layers:
            raise RuntimeError("YOLO model 에 사용 가능한 backbone+neck 이 없습니다.")
        self.layers = nn.ModuleList(layers)

        # ---- 모든 파라미터 동결, eval 모드 (BN running stat 고정) ----
        for p in self.parameters():
            p.requires_grad = False
        super().eval()

        # ---- dummy forward 로 layer 별 output shape 기록 ----
        self.input_size = int(input_size)
        layer_info = self._trace_layer_shapes(self.input_size)
        self.layer_info: List[Tuple[int, str, Optional[Tuple[int, ...]]]] = layer_info

        # ---- P3/P4/P5 자동 탐지 (각 stride 의 *마지막* layer index) ----
        target = {
            self.input_size // 8:  "p3",
            self.input_size // 16: "p4",
            self.input_size // 32: "p5",
        }
        chosen: Dict[str, Optional[int]] = {"p3": None, "p4": None, "p5": None}
        for (i, _name, shp) in layer_info:
            if shp is None or len(shp) != 4:
                continue
            h = shp[2]
            key = target.get(h)
            if key is not None:
                chosen[key] = i  # last occurrence wins
        if any(v is None for v in chosen.values()):
            raise RuntimeError(
                f"P3/P4/P5 자동 탐지 실패: {chosen}.  "
                f"input_size={self.input_size} 가 32 의 배수인지 확인."
            )
        self.p3_idx: int = int(chosen["p3"])  # type: ignore[arg-type]
        self.p4_idx: int = int(chosen["p4"])  # type: ignore[arg-type]
        self.p5_idx: int = int(chosen["p5"])  # type: ignore[arg-type]

        # ---- 선택된 P3/P4/P5 의 채널 수 캐싱 (Loss / debug 출력용) ----
        info_by_idx = {i: shp for (i, _n, shp) in layer_info if shp is not None}
        self.p3_shape = info_by_idx[self.p3_idx]
        self.p4_shape = info_by_idx[self.p4_idx]
        self.p5_shape = info_by_idx[self.p5_idx]

        if debug:
            self.print_layer_info()

    # ------------------------------------------------------------------
    # PyTorch hooks
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "YOLOFeatureExtractor":  # type: ignore[override]
        """항상 eval 모드 유지 — 부모 모델의 ``train(True)`` 가 BN 통계를
        오염시키지 않도록 차단."""
        return super().train(False)

    # ------------------------------------------------------------------
    # Internal: dummy forward to trace shapes
    # ------------------------------------------------------------------
    def _trace_layer_shapes(
        self, input_size: int,
    ) -> List[Tuple[int, str, Optional[Tuple[int, ...]]]]:
        """모든 layer 출력의 shape 를 기록 후 ``[(i, type_name, shape), ...]`` 반환."""
        info: List[Tuple[int, str, Optional[Tuple[int, ...]]]] = []
        device = next(self.parameters()).device
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        with torch.no_grad():
            self._forward_internal(dummy, collect_into=info)
        return info

    # ------------------------------------------------------------------
    # Internal: ultralytics-style forward (handles skip-connection f)
    # ------------------------------------------------------------------
    def _forward_internal(
        self,
        x: torch.Tensor,
        collect_into: Optional[List[Tuple[int, str, Optional[Tuple[int, ...]]]]] = None,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """단일 forward pass 실행.

        Parameters
        ----------
        x : Tensor
            입력 텐서 (B, 3, H, W), 값 범위 [0, 1].
        collect_into : list | None
            지정되면 (layer_idx, type_name, shape) 를 append.

        Returns
        -------
        (final_x, saved_outputs)
            ``saved_outputs[i]`` = i 번째 layer 출력.
        """
        y: List[Any] = []
        for i, m in enumerate(self.layers):
            f = getattr(m, "f", -1)
            if f != -1:
                # ultralytics 의 분기 — int 면 단일 source, list 면 Concat 입력
                if isinstance(f, int):
                    xx = y[f]
                else:
                    xx = [x if j == -1 else y[j] for j in f]
            else:
                xx = x
            x = m(xx)
            y.append(x)
            if collect_into is not None:
                shp = tuple(x.shape) if isinstance(x, torch.Tensor) else None
                collect_into.append((i, type(m).__name__, shp))
        return x, y

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, x_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``x_01`` ∈ [0, 1], shape (B, 3, H, W) → P3/P4/P5 feature dict.

        반환 dict: ``{"p3": Tensor[B, C3, H/8,  W/8],
                     "p4": Tensor[B, C4, H/16, W/16],
                     "p5": Tensor[B, C5, H/32, W/32]}``.

        ``torch.no_grad`` 로 감싸지 *않는다*.  caller (DetectionAwareLoss) 가
        LUNA branch 일 때 grad 가 그대로 흘러야 하므로.
        """
        feats: Dict[str, torch.Tensor] = {}
        y: List[Any] = []
        for i, m in enumerate(self.layers):
            f = getattr(m, "f", -1)
            if f != -1:
                if isinstance(f, int):
                    xx = y[f]
                else:
                    xx = [x_01 if j == -1 else y[j] for j in f]
            else:
                xx = x_01
            x_01 = m(xx)
            y.append(x_01)
            if i == self.p3_idx:
                feats["p3"] = x_01
            elif i == self.p4_idx:
                feats["p4"] = x_01
            elif i == self.p5_idx:
                feats["p5"] = x_01
        return feats

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def print_layer_info(self) -> None:
        """layer index / type / output shape 표 + P3/P4/P5 선택 결과 출력."""
        print("-" * 86)
        print(f"  YOLOv8n layer trace  (input = 1×3×{self.input_size}×{self.input_size})")
        print("-" * 86)
        print(f"  {'idx':>4}  {'type':<14}  {'output shape':<28}  picked")
        print("-" * 86)
        for (i, name, shp) in self.layer_info:
            shp_str = "x".join(str(s) for s in shp) if shp else "(None)"
            tag = ""
            if   i == self.p3_idx: tag = "← P3"
            elif i == self.p4_idx: tag = "← P4"
            elif i == self.p5_idx: tag = "← P5"
            print(f"  {i:>4}  {name:<14}  {shp_str:<28}  {tag}")
        print("-" * 86)
        print(f"  Selected P3 idx={self.p3_idx} shape={self.p3_shape}")
        print(f"  Selected P4 idx={self.p4_idx} shape={self.p4_shape}")
        print(f"  Selected P5 idx={self.p5_idx} shape={self.p5_shape}")
        print("-" * 86)

    def feature_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """``{"p3": shape, "p4": shape, "p5": shape}`` — Loss 초기화 디버그용."""
        return {"p3": self.p3_shape, "p4": self.p4_shape, "p5": self.p5_shape}
