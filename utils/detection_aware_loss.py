"""Detection-aware loss — frozen YOLOv8n multi-level feature 정합 loss.

수식
----
    L_detection_aware = L_feat_mse + α · L_feat_cos + β · L_preserve

각 항
^^^^^^
* **L_feat_mse**  : LUNA 출력과 GT 의 P3/P4/P5 feature MSE 합.
    feature magnitude 정합 — "비슷한 활성도" 를 만들기.
* **L_feat_cos**  : 각 level 의 (B, C, H·W) feature 를 채널 차원에서 코사인 유사도
    → ``1 - cos`` 평균. feature *방향* 정합 — magnitude 가 달라도 패턴이 같으면
    낮은 loss. MSE 단독의 한계 (magnitude 만 맞추는 trivial solution) 를 보완.
* **L_preserve** : LUNA 출력과 *원본 low* 의 P3/P4/P5 feature MSE 합 (optional,
    ``use_preserve=True`` 일 때만).  YOLO 가 이미 low 에서 잘 검출하던 영역의
    feature 를 LUNA 가 망가뜨리지 못하게 잡아주는 anchor.  단, low 의 *나쁜*
    feature (노이즈 / 미검출) 까지 보존할 위험이 있어 기본은 off.

설계 디테일
-----------
* level 별 weight (``level_weights``): 기본 P3=P4=P5=1.0.
* 각 level loss 는 ``F.mse_loss(..., reduction='mean')`` / ``F.cosine_similarity``
  의 ``.mean()`` 으로 spatial / channel 크기에 무관하게 ``O(1)`` scale.
* GT branch (``feat_high``) 와 low branch (``feat_low``) 의 forward 는
  ``torch.no_grad`` 로 감싼다 — 메모리 절감 + gradient 가 LUNA 까지 흐를 필요
  없음.  LUNA branch (``feat_luna``) 의 forward 는 grad 필요 → 일반 forward.

입력 가정
---------
LUNA generator 출력 / dataset GT 모두 ``[-1, 1]`` 범위.  Loss 내부에서
``(x + 1) / 2`` 변환 후 ``clamp(0, 1)`` 로 YOLO 입력 도메인에 맞춤.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.yolo_features import YOLOFeatureExtractor


LEVELS = ("p3", "p4", "p5")


class DetectionAwareLoss(nn.Module):
    """L_feat_mse + α · L_feat_cos + β · L_preserve (preserve optional).

    Parameters
    ----------
    yolo_extractor : YOLOFeatureExtractor
        Frozen YOLOv8n P3/P4/P5 추출기.  외부에서 한 번 만들고 주입 (싱글톤).
    alpha_cos : float
        Cosine similarity loss 가중치 (default 0.5).
    beta_preserve : float
        Preserve loss 가중치 (default 0.1, ``use_preserve=True`` 일 때만).
    use_preserve : bool
        Low image branch feature 보존 loss 사용 여부 (default False).
    level_weights : dict | None
        ``{"p3": w3, "p4": w4, "p5": w5}``.  None 이면 모두 1.0.
    """

    def __init__(
        self,
        yolo_extractor: YOLOFeatureExtractor,
        alpha_cos: float = 0.5,
        beta_preserve: float = 0.1,
        use_preserve: bool = False,
        level_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.yolo = yolo_extractor
        self.alpha_cos = float(alpha_cos)
        self.beta_preserve = float(beta_preserve)
        self.use_preserve = bool(use_preserve)
        if level_weights is None:
            level_weights = {"p3": 1.0, "p4": 1.0, "p5": 1.0}
        # 누락된 key 는 0 으로 (사실상 비활성)
        self.level_weights = {k: float(level_weights.get(k, 0.0)) for k in LEVELS}

    # ------------------------------------------------------------------
    @staticmethod
    def _to_01(x_pm1: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → [0, 1] 변환 + 안전 clamp."""
        return ((x_pm1 + 1.0) * 0.5).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_level(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """단일 level cosine similarity → ``1 - mean(cos_sim)``.

        ``a, b`` : (B, C, H, W).  채널 축을 따라 spatial flatten 후 코사인.
        각 (B, position) 마다 C 차원 벡터의 코사인을 구하고 평균.
        """
        # (B, C, H*W) — channel 축이 비교 단위가 되도록 dim=1 사용
        a_f = a.flatten(2)
        b_f = b.flatten(2)
        cos = F.cosine_similarity(a_f, b_f, dim=1)  # → (B, H*W)
        return 1.0 - cos.mean()

    # ------------------------------------------------------------------
    def forward(
        self,
        luna_pm1: torch.Tensor,
        high_pm1: torch.Tensor,
        low_pm1: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """detection-aware loss 계산.

        Parameters
        ----------
        luna_pm1 : Tensor
            Generator 출력 ``[-1, 1]``, shape (B, 3, H, W).
        high_pm1 : Tensor
            Ground-truth normal-light 이미지, 같은 shape.
        low_pm1 : Tensor | None
            원본 low image (preserve loss 용).  ``use_preserve=True`` 이면 필수.

        Returns
        -------
        dict[str, Tensor]
            ``{
                "total":     L_feat_mse + α·L_feat_cos + β·L_preserve,
                "feat_mse":  Tensor (detach),
                "feat_cos":  Tensor (detach),
                "preserve":  Tensor (detach, preserve 비활성 시 0),
                "per_level_mse": {"p3": ..., "p4": ..., "p5": ...},
                "per_level_cos": {"p3": ..., "p4": ..., "p5": ...},
              }``.
        """
        device = luna_pm1.device

        # ---- LUNA branch (gradient 필요) ----
        luna_01 = self._to_01(luna_pm1)
        feat_luna = self.yolo(luna_01)

        # ---- GT branch (target, no grad) ----
        with torch.no_grad():
            feat_high = self.yolo(self._to_01(high_pm1))
        # 안전: 모든 target 텐서 detach
        feat_high = {k: v.detach() for k, v in feat_high.items()}

        # ---- Low branch (preserve target, no grad) ----
        feat_low: Optional[Dict[str, torch.Tensor]] = None
        if self.use_preserve:
            if low_pm1 is None:
                raise ValueError("use_preserve=True 이면 low_pm1 인자가 필요합니다.")
            with torch.no_grad():
                feat_low_raw = self.yolo(self._to_01(low_pm1))
            feat_low = {k: v.detach() for k, v in feat_low_raw.items()}

        # ---- Level 별 loss ----
        per_mse: Dict[str, torch.Tensor] = {}
        per_cos: Dict[str, torch.Tensor] = {}
        per_pres: Dict[str, torch.Tensor] = {}
        for level in LEVELS:
            w = self.level_weights[level]
            f_l = feat_luna[level]
            f_h = feat_high[level]
            per_mse[level] = w * F.mse_loss(f_l, f_h)
            per_cos[level] = w * self._cosine_level(f_l, f_h)
            if self.use_preserve and feat_low is not None:
                per_pres[level] = w * F.mse_loss(f_l, feat_low[level])

        l_feat_mse = sum(per_mse.values())
        l_feat_cos = sum(per_cos.values())
        l_preserve = (
            sum(per_pres.values()) if self.use_preserve and per_pres
            else torch.zeros((), device=device)
        )

        total = l_feat_mse + self.alpha_cos * l_feat_cos
        if self.use_preserve:
            total = total + self.beta_preserve * l_preserve

        return {
            "total":    total,
            "feat_mse": l_feat_mse.detach(),
            "feat_cos": l_feat_cos.detach(),
            "preserve": l_preserve.detach() if isinstance(l_preserve, torch.Tensor)
                        else torch.tensor(float(l_preserve), device=device),
            "per_level_mse": {k: v.detach() for k, v in per_mse.items()},
            "per_level_cos": {k: v.detach() for k, v in per_cos.items()},
        }
