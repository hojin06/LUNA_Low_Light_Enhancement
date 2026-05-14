"""Paired augmentation — LOL 페어(low ↔ high) 동기화 변환.

설계 의도 (Design rationale)
---------------------------
* **페어 어긋남 방지**: low 와 high 는 같은 장면의 다른 조도 이미지이므로 모든
  기하학적 변환(crop / flip / rotate / perspective) 은 **동일한 난수 파라미터로
  두 이미지에 적용**되어야 한다. 본 클래스는 변환 직전에 파라미터를 추출한 뒤
  `torchvision.transforms.functional` 의 결정론적 API 로 두 이미지에 차례로
  적용한다.

* **Low-viewpoint simulation**: 전기 벨로모빌의 30~50 cm 시점 카메라 환경을
  모사하기 위해 (a) 하단 60~70 % 만 추출하여 확대하는 perspective transform 과
  (b) 노면 영역(하반부) 비중을 높이는 비대칭 crop 을 추가한다. 학술 논문의
  "Low-viewpoint simulation augmentation" contribution 의 근거 모듈.

* **저조도 강화 (low 전용)**: low 이미지에만 gamma 증가, 밝기 감소, Gaussian
  noise 를 확률적으로 추가한다. high(GT) 의 의미적 정합성을 깨지 않으면서
  더 어두운 야간 / 저가 센서 환경을 시뮬레이션.

* 외부 의존성 없음(torchvision + PIL 만 사용) — 임베디드/CI 재현성 용이.
"""
from __future__ import annotations

import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

# 페어 변환 입출력 별칭
PairTensor = Tuple[torch.Tensor, torch.Tensor]
PairPIL = Tuple[Image.Image, Image.Image]


class PairedAugment:
    """LOL (low, high) 페어 동기화 + low 전용 저조도 강화.

    Parameters
    ----------
    image_size : int
        출력 해상도 (정사각). 기본 256.
    training : bool
        True 면 모든 augmentation 활성, False 면 resize 만 적용.

    확률 / 범위 인자는 모두 키워드로 노출되어 ablation 실험에서 손쉽게 조절 가능.

    Returns
    -------
    (low, high) : tuple[Tensor, Tensor]
        각각 ``(3, image_size, image_size)``, 값 범위 ``[-1, 1]``.
    """

    def __init__(
        self,
        image_size: int = 256,
        training: bool = True,
        full_resize: bool = False,
        # ---- Geometric (paired) ----
        p_flip: float = 0.5,
        p_rotate: float = 0.5,
        rotate_deg: float = 10.0,
        crop_scale: Tuple[float, float] = (0.7, 1.0),
        p_lowerhalf_crop: float = 0.5,
        p_perspective: float = 0.4,
        perspective_keep: Tuple[float, float] = (0.6, 0.7),
        # ---- Photometric (low only) ----
        p_gamma: float = 0.4,
        gamma_range: Tuple[float, float] = (1.5, 3.0),
        p_brightness: float = 0.4,
        brightness_range: Tuple[float, float] = (0.3, 0.7),
        p_noise: float = 0.5,
        noise_sigma_range: Tuple[float, float] = (0.01, 0.05),
    ) -> None:
        self.image_size = image_size
        self.training = training
        self.full_resize = full_resize
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.rotate_deg = rotate_deg
        self.crop_scale = crop_scale
        self.p_lowerhalf_crop = p_lowerhalf_crop
        self.p_perspective = p_perspective
        self.perspective_keep = perspective_keep
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_brightness = p_brightness
        self.brightness_range = brightness_range
        self.p_noise = p_noise
        self.noise_sigma_range = noise_sigma_range

    # ==================================================================
    # Public entry
    # ==================================================================
    def __call__(self, low_img: Image.Image, high_img: Image.Image) -> PairTensor:
        if not self.training:
            return self._eval_path(low_img, high_img)
        return self._train_path(low_img, high_img)

    # ==================================================================
    # Eval path: resize only (no augmentation)
    # ==================================================================
    def _eval_path(self, low_img: Image.Image, high_img: Image.Image) -> PairTensor:
        size = [self.image_size, self.image_size]
        low_img = TF.resize(low_img, size, interpolation=InterpolationMode.BILINEAR)
        high_img = TF.resize(high_img, size, interpolation=InterpolationMode.BILINEAR)
        return self._to_norm_tensor(low_img), self._to_norm_tensor(high_img)

    # ==================================================================
    # Train path
    # ==================================================================
    def _train_path(self, low_img: Image.Image, high_img: Image.Image) -> PairTensor:
        # ---- 1. Paired geometric on PIL ----
        if self.full_resize:
            # 전체 이미지 리사이즈 (random crop 생략). 광시야 / 컨텍스트 보존 우선.
            size = [self.image_size, self.image_size]
            low_img = TF.resize(low_img, size, interpolation=InterpolationMode.BILINEAR)
            high_img = TF.resize(high_img, size, interpolation=InterpolationMode.BILINEAR)
        else:
            low_img, high_img = self._paired_crop_resize(low_img, high_img)
        low_img, high_img = self._paired_hflip(low_img, high_img)
        low_img, high_img = self._paired_rotate(low_img, high_img)
        low_img, high_img = self._paired_perspective(low_img, high_img)

        # ---- 2. PIL → Tensor [0, 1] ----
        low_t = TF.to_tensor(low_img)
        high_t = TF.to_tensor(high_img)

        # ---- 3. Photometric (low only) ----
        low_t = self._photometric_low_only(low_t)

        # ---- 4. [0, 1] → [-1, 1] ----
        return low_t * 2.0 - 1.0, high_t * 2.0 - 1.0

    # ==================================================================
    # Paired geometric operations
    # ==================================================================
    def _paired_crop_resize(
        self, low: Image.Image, high: Image.Image
    ) -> PairPIL:
        """랜덤 crop + 256×256 resize. 50 % 확률로 하단 편향 크롭."""
        W, H = low.size  # PIL: (width, height)
        scale = random.uniform(*self.crop_scale)
        crop_h = max(int(H * scale), 16)
        crop_w = max(int(W * scale), 16)

        if random.random() < self.p_lowerhalf_crop:
            # 노면 영역 강조: top 좌표를 하단 쪽으로 편향
            lo = int((H - crop_h) * 0.3)
            hi = max(H - crop_h, lo)
            top = random.randint(lo, hi) if hi > lo else 0
        else:
            top = random.randint(0, max(H - crop_h, 0))
        left = random.randint(0, max(W - crop_w, 0))

        size = [self.image_size, self.image_size]
        low = TF.resized_crop(low, top, left, crop_h, crop_w, size,
                              interpolation=InterpolationMode.BILINEAR)
        high = TF.resized_crop(high, top, left, crop_h, crop_w, size,
                               interpolation=InterpolationMode.BILINEAR)
        return low, high

    def _paired_hflip(self, low: Image.Image, high: Image.Image) -> PairPIL:
        if random.random() < self.p_flip:
            return TF.hflip(low), TF.hflip(high)
        return low, high

    def _paired_rotate(self, low: Image.Image, high: Image.Image) -> PairPIL:
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            low = TF.rotate(low, angle,
                            interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])
            high = TF.rotate(high, angle,
                             interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])
        return low, high

    def _paired_perspective(self, low: Image.Image, high: Image.Image) -> PairPIL:
        """저시점 perspective: 이미지 하단 60~70 % 영역을 전체로 stretch.

        startpoints (원본에서 샘플링할 trapezoid) → endpoints (출력 전체 사각형).
        narrow 항으로 약간의 사다리꼴 폭 축소를 더해 perspective 느낌을 강화.
        """
        if random.random() >= self.p_perspective:
            return low, high

        s = self.image_size
        keep = random.uniform(*self.perspective_keep)  # 0.60 ~ 0.70
        top_y = int(s * (1.0 - keep))
        narrow = int(random.uniform(0.0, 0.15) * s)

        # 순서: [top-left, top-right, bottom-right, bottom-left]
        startpoints = [
            [narrow, top_y],
            [s - narrow, top_y],
            [s, s],
            [0, s],
        ]
        endpoints = [
            [0, 0],
            [s, 0],
            [s, s],
            [0, s],
        ]
        low = TF.perspective(low, startpoints, endpoints,
                             interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])
        high = TF.perspective(high, startpoints, endpoints,
                              interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])
        return low, high

    # ==================================================================
    # Low-only photometric augmentation
    # ==================================================================
    def _photometric_low_only(self, low_t: torch.Tensor) -> torch.Tensor:
        """gamma ↑ / brightness ↓ / Gaussian noise — 더 어두운 환경 시뮬레이션."""
        # 1) Gamma 변환 (x^γ, γ > 1 이면 어두워짐)
        if random.random() < self.p_gamma:
            gamma = random.uniform(*self.gamma_range)
            low_t = low_t.clamp(min=1e-6).pow(gamma)

        # 2) 밝기 감소 (multiplicative)
        if random.random() < self.p_brightness:
            factor = random.uniform(*self.brightness_range)
            low_t = low_t * factor

        # 3) 가우시안 노이즈 (저가 CMOS 센서 노이즈 모사)
        if random.random() < self.p_noise:
            sigma = random.uniform(*self.noise_sigma_range)
            low_t = low_t + torch.randn_like(low_t) * sigma

        return low_t.clamp(0.0, 1.0)

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _to_norm_tensor(img: Image.Image) -> torch.Tensor:
        """PIL → tensor [0,1] → [-1, 1]."""
        return TF.to_tensor(img) * 2.0 - 1.0
