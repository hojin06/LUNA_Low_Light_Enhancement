"""LightEnhanceGenerator — 저조도 이미지 향상용 경량 U-Net Generator.

설계 근거 (Design rationale)
----------------------------
* **U-Net 4-stage 뼈대**: FUnIE-GAN [Islam et al., 2020, IEEE RA-L]의
  encoder–decoder + skip connection 구조를 차용. 본 모델은 저시점 카메라
  (≤50 cm 노면 시점)의 fine-grained texture 보존이 필수이므로 skip
  connection으로 저수준 디테일을 디코더에 직접 전달한다.
* **Depthwise Separable Convolution (DSConv)**: Zero-DCE++ [Li et al., 2021,
  IEEE TPAMI]를 따라 모든 spatial conv를 depthwise + pointwise로 분해.
  표준 3×3 conv 대비 약 8~9배 적은 MAC 비용으로 임베디드(Jetson Orin Nano,
  15 W) 30 FPS 추론을 달성한다.
* **Lightweight Attention (CA + SA)**: 채널/공간 attention을 bottleneck에
  삽입. EnlightenGAN [Jiang et al., 2021, IEEE TIP]의 illumination-aware
  attention 아이디어를 CBAM [Woo et al., 2018, ECCV]의 직렬 구조로 단순화.
  저조도 영역(어두운 픽셀)에 더 큰 가중치를 주는 효과를 학습으로 유도한다.

채널 구성 (base_filters=24): 3 → 24 → 48 → 96 → 192 → 96 → 48 → 24 → 3.
입출력 범위: [-1, 1] (Tanh).
"""
from __future__ import annotations

from typing import Dict, Optional  # noqa: F401  (forward-ref 문자열에서 사용)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DSConv(nn.Module):
    """Depthwise Separable Convolution block.

    구조: ``DW-Conv3x3 → BN → ReLU → PW-Conv1x1 → BN → ReLU``.

    - depthwise: ``groups = in_channels`` 이므로 채널별 독립 spatial filter
    - pointwise: 1×1 conv로 채널 혼합 (cross-channel projection)

    표준 3×3 conv 대비 MAC 절감 비율 ≈ ``1/c_out + 1/9``. (MobileNetV1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        # Depthwise (spatial filtering)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=not use_bn,
        )
        self.bn1 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        # Pointwise (channel mixing)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=not use_bn
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class ConvBlock(nn.Module):
    """표준 Conv 블록 — DSConv 의 ablation 대조군 (use_dsconv=False).

    구조: ``Conv3x3 → BN → ReLU``. DSConv 와 동일한 ``(c_in, c_out, stride)``
    인터페이스를 유지하므로 architecture 그대로 두고 블록만 swap 가능.
    파라미터 수는 약 8~9× 증가하여 DSConv 의 경량성 효과를 정량 비교할 때 사용.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=not use_bn,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """Channel Attention Module (CBAM, Woo et al., 2018).

    AvgPool 와 MaxPool 의 채널 디스크립터를 공유 MLP에 통과시킨 뒤 합산하여
    sigmoid 게이팅을 적용. 저조도 환경에서 정보량이 풍부한 채널(보통 G/B
    저주파 성분)을 강조하는 효과를 학습한다.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        # 1×1 conv pair는 FC 두 개와 등가이지만 텐서 reshape이 필요 없어 효율적
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM, Woo et al., 2018).

    채널 축으로 average / max pooling 한 2-채널 맵을 7×7 conv로 결합 후
    sigmoid. "어디(where)"에 주목할지를 학습한다. 저조도에서는 밝은 가로등
    주변과 짙은 음영 경계가 강조될 것으로 기대.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel_size must be 3, 5, or 7"
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class LightAttention(nn.Module):
    """Lightweight Attention = ChannelAttention → SpatialAttention.

    Bottleneck (가장 깊은 32×32 feature)에 삽입. 추가 파라미터는 약 9 K로
    전체 대비 10 % 수준이며 receptive field 가장 큰 위치에서 최대 효율.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class LightEnhanceGenerator(nn.Module):
    """저조도 이미지 향상용 4-stage DSConv U-Net Generator.

    Parameters
    ----------
    in_channels : int
        입력 채널 수 (RGB = 3).
    out_channels : int
        출력 채널 수 (RGB = 3).
    base_filters : int
        첫 번째 인코더 stage의 채널 수. 이후 ×2 배율로 증가.

    Shapes (입력 256×256 기준)
    --------------------------
    | Stage  | Op                | Shape           |
    |--------|-------------------|-----------------|
    | input  |                   | 3 ×256×256     |
    | enc1   | DSConv 3→24, s=1  | 24 ×256×256    |
    | enc2   | DSConv 24→48, s=2 | 48 ×128×128    |
    | enc3   | DSConv 48→96, s=2 | 96 ×64 ×64     |
    | enc4   | DSConv 96→192,s=2 | 192×32 ×32     |
    | attn   | CA + SA           | 192×32 ×32     |
    | dec3   | up + skip + DS    | 96 ×64 ×64     |
    | dec2   | up + skip + DS    | 48 ×128×128    |
    | dec1   | up + skip + DS    | 24 ×256×256    |
    | out    | Conv 1×1 + Tanh   | 3  ×256×256    |
    """

    # 9-블록 hybrid layout 의 conv_config 키 (output_conv 는 항상 1×1 Conv2d).
    HYBRID_BLOCK_NAMES = (
        "input_conv", "enc1", "enc2", "enc3", "bottleneck",
        "dec3", "dec2", "dec1",
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 32,
        use_attention: bool = True,
        use_dsconv: bool = True,
        conv_config: "Optional[Dict[str, str]]" = None,
    ) -> None:
        """Generator 초기화 — 두 layout 지원.

        * ``conv_config = None`` (default): legacy 8-블록 layout
          (``enc1..4 + dec3..1 + out_proj``).  기존 체크포인트와 호환.
        * ``conv_config`` 지정 시: 9-블록 hybrid layout
          (``input_conv → enc1..3 → bottleneck → dec3..1 → output_conv``).
          각 블록을 'dsconv' / 'standard' 로 개별 지정 가능. ``output_conv``
          는 1×1 plain Conv 로 고정.
        """
        super().__init__()
        c1 = base_filters
        c2 = base_filters * 2
        c3 = base_filters * 4
        c4 = base_filters * 8

        # 공통 메타데이터
        self._base_filters = base_filters
        self._use_attention = use_attention
        self._use_dsconv = use_dsconv
        self._conv_config = conv_config
        self._layout = "hybrid" if conv_config is not None else "legacy"

        if conv_config is None:
            # ---------- Legacy 8-블록 layout ----------
            block = DSConv if use_dsconv else ConvBlock

            self.enc1 = block(in_channels, c1, stride=1)  # 256, skip1
            self.enc2 = block(c1, c2, stride=2)           # 128, skip2
            self.enc3 = block(c2, c3, stride=2)           # 64 , skip3
            self.enc4 = block(c3, c4, stride=2)           # 32 , bottleneck

            self.attn = LightAttention(c4) if use_attention else nn.Identity()

            self.dec3 = block(c4 + c3, c3, stride=1)
            self.dec2 = block(c3 + c2, c2, stride=1)
            self.dec1 = block(c2 + c1, c1, stride=1)

            self.out_proj = nn.Conv2d(c1, out_channels, kernel_size=1)
        else:
            # ---------- Hybrid 9-블록 layout ----------
            self._validate_conv_config(conv_config)

            def _block(name: str, c_in: int, c_out: int, stride: int = 1) -> nn.Module:
                kind = conv_config.get(name, "dsconv")
                if kind == "dsconv":
                    return DSConv(c_in, c_out, stride=stride)
                return ConvBlock(c_in, c_out, stride=stride)

            self.input_conv  = _block("input_conv", in_channels, c1, stride=1)  # 256
            self.enc1        = _block("enc1", c1, c2, stride=2)                 # 128
            self.enc2        = _block("enc2", c2, c3, stride=2)                 # 64
            self.enc3        = _block("enc3", c3, c4, stride=2)                 # 32
            self.bottleneck  = _block("bottleneck", c4, c4, stride=1)           # 32

            self.attn = LightAttention(c4) if use_attention else nn.Identity()

            self.dec3 = _block("dec3", c4 + c3, c3, stride=1)
            self.dec2 = _block("dec2", c3 + c2, c2, stride=1)
            self.dec1 = _block("dec1", c2 + c1, c1, stride=1)

            # output_conv 는 항상 plain 1×1 Conv (BN 없음, Tanh 직전)
            self.output_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    @classmethod
    def _validate_conv_config(cls, cfg: Dict[str, str]) -> None:
        missing = [k for k in cls.HYBRID_BLOCK_NAMES if k not in cfg]
        if missing:
            raise ValueError(
                f"conv_config 누락 키: {missing}.  필요한 키: {list(cls.HYBRID_BLOCK_NAMES)}"
            )
        for name, kind in cfg.items():
            if name not in cls.HYBRID_BLOCK_NAMES:
                raise ValueError(f"conv_config 의 알 수 없는 블록 이름: '{name}'")
            if kind not in ("dsconv", "standard"):
                raise ValueError(
                    f"conv_config['{name}'] 는 'dsconv' 또는 'standard' 여야 합니다 (got '{kind}')"
                )

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming initialization (저조도 입력의 작은 dynamic range 보완)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        """ Bilinear ×2 up-sampling (transpose-conv 대비 체커보드 artefact 없음)."""
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._layout == "legacy":
            return self._forward_legacy(x)
        return self._forward_hybrid(x)

    def _forward_legacy(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)   # c1 × 256
        s2 = self.enc2(s1)  # c2 × 128
        s3 = self.enc3(s2)  # c3 × 64
        s4 = self.enc4(s3)  # c4 × 32
        b = self.attn(s4)
        u3 = self.dec3(torch.cat([self._up(b), s3], dim=1))
        u2 = self.dec2(torch.cat([self._up(u3), s2], dim=1))
        u1 = self.dec1(torch.cat([self._up(u2), s1], dim=1))
        return torch.tanh(self.out_proj(u1))

    def _forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.input_conv(x)   # c1 × 256, skip → dec1
        s1 = self.enc1(s0)        # c2 × 128, skip → dec2
        s2 = self.enc2(s1)        # c3 × 64 , skip → dec3
        s3 = self.enc3(s2)        # c4 × 32
        b = self.bottleneck(s3)   # c4 × 32  ← 추가된 채널-혼합 블록
        b = self.attn(b)          # c4 × 32  (CA + SA 또는 Identity)
        u3 = self.dec3(torch.cat([self._up(b),  s2], dim=1))  # c3 × 64
        u2 = self.dec2(torch.cat([self._up(u3), s1], dim=1))  # c2 × 128
        u1 = self.dec1(torch.cat([self._up(u2), s0], dim=1))  # c1 × 256
        return torch.tanh(self.output_conv(u1))
