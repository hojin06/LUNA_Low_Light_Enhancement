"""PatchGAN Discriminator — Markovian patch-level real/fake classifier.

설계 근거 (Design rationale)
----------------------------
* **Markovian PatchGAN**: pix2pix [Isola et al., 2017, CVPR] 및 FUnIE-GAN
  [Islam et al., 2020, RA-L]의 70×70 PatchGAN 구조를 단순화. 이미지 전체에
  단일 점수가 아닌 16×16 patch 별 점수 맵을 출력해 국소(local) 사실성을
  강조. 도로 표지·차선 등 작은 구조 보존이 중요한 자율주행 시나리오에 적합.
* **InstanceNorm + LeakyReLU**: 작은 batch size(현장 학습 batch=4~8)에서
  안정적이며 GAN 학습 표준 조합.
* **Conditional input**: 원본 저조도 이미지(``x``)와 향상 이미지(``y_hat``)
  를 채널 축으로 concat 한 6-채널 입력 → Generator가 "원본과 정합되는
  향상"을 학습하도록 강제 (conditional GAN, cGAN 형식).

추론 시에는 사용되지 않으므로 모델 배포 페이로드에서 제외 가능.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """4-층 stride-2 Conv 로 16×16 patch 점수 맵을 출력하는 Markovian PatchGAN.

    채널 구성: 6 → 32 → 64 → 128 → 256 → 1 (logits).
    출력은 ``BCEWithLogitsLoss`` 와 함께 사용하므로 sigmoid 미적용.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_filters: int = 32,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        c1 = base_filters
        c2 = base_filters * 2
        c3 = base_filters * 4
        c4 = base_filters * 8

        def block(c_in: int, c_out: int, norm: bool = True) -> nn.Sequential:
            """Conv(4×4, s=2) → [InstanceNorm] → LeakyReLU(0.2)."""
            layers: List[nn.Module] = [
                nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not norm,
                )
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(c_out, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_channels, c1, norm=False),  # 256 → 128
            block(c1, c2),                       # 128 → 64
            block(c2, c3),                       # 64  → 32
            block(c3, c4),                       # 32  → 16
            # 3×3 conv (stride 1) → patch-wise logits, padding=1 로 shape 유지
            nn.Conv2d(c4, 1, kernel_size=3, stride=1, padding=1),
        )

        self._init_weights()

        # ---- Spectral Normalization (Miyato et al., 2018, ICLR) ----
        # D 의 Lipschitz 상수를 1 로 제약 → 학습 안정성↑, mode collapse 완화.
        # init 이후 적용해야 parametrization 이 정상 초기 weight 를 받음.
        self.use_spectral_norm = use_spectral_norm
        if use_spectral_norm:
            try:
                from torch.nn.utils.parametrizations import spectral_norm
            except ImportError:  # 구버전 torch fallback
                from torch.nn.utils import spectral_norm  # type: ignore[no-redef]
            for m in self.net.modules():
                if isinstance(m, nn.Conv2d):
                    spectral_norm(m)

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """DCGAN 권고 초기화: N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        x : torch.Tensor
            shape ``(B, 6, 256, 256)``  — 원본 저조도와 향상 이미지를 concat.

        Returns
        -------
        torch.Tensor
            shape ``(B, 1, 16, 16)`` — patch-wise real/fake logits.
        """
        return self.net(x)
