"""GAN 학습 손실 함수 모음 (CombinedLoss = Adv + L1 + VGG + SSIM).

설계 근거 (Design rationale)
----------------------------
* **L1 (global similarity)**: FUnIE-GAN [Islam et al., 2020, RA-L]은
  pixel-wise L1 으로 global color/luminance 정합을 유도. λ_L1 = 0.7.
* **VGG perceptual loss (content loss)**: FUnIE-GAN 동일 — VGG16 relu3_3
  특징 공간의 L1 거리. 의미적 구조(차선, 노면 텍스처)를 보존. λ_VGG = 0.3.
* **SSIM loss (본 논문 추가 contribution)**: pixel-wise loss 는 평균적
  품질에 강하나 국소 구조 보존이 약함 → 다중 채널 single-scale SSIM
  [Wang et al., 2004, IEEE TIP] 으로 패치 단위 luminance/contrast/
  structure 유사도를 보조 신호로 추가. λ_SSIM = 0.5.
* **Adversarial (BCE with logits)**: vanilla GAN. LSGAN/Hinge 변형이
  필요하면 `adv_mode` 인자로 확장.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Perceptual loss (VGG16 relu3_3 feature L1)
# ---------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """VGG16 relu3_3 feature-space L1 distance.

    입력 가정: ``[-1, 1]`` 범위의 Generator 출력. 내부에서 ``[0, 1]`` 로 역정
    규화한 뒤 ImageNet 평균/표준편차로 normalize 하여 pretrained VGG 통과.

    Notes
    -----
    - ``features[:16]`` 까지 추출 → index 15 가 ``relu3_3``.
      (VGG16 features: Conv–ReLU–Conv–ReLU–Pool–…)
    - VGG 파라미터는 freeze (``requires_grad=False``) 후 eval 모드.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import vgg16

        try:
            # Newer torchvision API (≥ 0.13)
            from torchvision.models import VGG16_Weights  # type: ignore

            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            # Fallback: 인터넷이 없거나 구버전 torchvision
            try:
                vgg = vgg16(pretrained=True)
            except Exception:
                vgg = vgg16(weights=None)  # 가중치 미로딩 (warning)
                print(
                    "[PerceptualLoss] WARNING: VGG16 pretrained weights not "
                    "available. Using random init — perceptual loss will be "
                    "uninformative until weights are loaded."
                )

        features = vgg.features[:16].eval()
        for p in features.parameters():
            p.requires_grad = False
        self.features = features

        # ImageNet normalization stats
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → [0, 1] → ImageNet normalize."""
        x01 = (x + 1.0) * 0.5
        return (x01 - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f_pred = self.features(self._prepare(pred))
        f_targ = self.features(self._prepare(target))
        return F.l1_loss(f_pred, f_targ)


# ---------------------------------------------------------------------------
# SSIM loss
# ---------------------------------------------------------------------------
class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) 기반 손실.

    Wang et al., 2004. Gaussian window 로 local statistics 추정 후
    luminance·contrast·structure 항목을 동시에 평가. 본 구현은 single-scale,
    multi-channel SSIM 의 평균을 사용 (MS-SSIM 대비 계산량 절감).

    입력 가정: ``[-1, 1]`` 범위. 내부에서 ``[0, 1]`` 로 변환.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 3,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.data_range = data_range

        # 분리 가능 Gaussian 을 2D 로 확장 후 채널별 grouped conv 로 적용
        kernel_1d = self._gaussian_window(window_size, sigma)
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # (k, k)
        kernel = kernel_2d.expand(channels, 1, window_size, window_size)
        self.register_buffer("window", kernel.contiguous())

        # 표준 SSIM 안정화 상수 (data_range 정규화 후)
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    @staticmethod
    def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        groups = self.channels

        mu_x = F.conv2d(x, self.window, padding=pad, groups=groups)
        mu_y = F.conv2d(y, self.window, padding=pad, groups=groups)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, self.window, padding=pad, groups=groups) - mu_x2
        sigma_y2 = F.conv2d(y * y, self.window, padding=pad, groups=groups) - mu_y2
        sigma_xy = F.conv2d(x * y, self.window, padding=pad, groups=groups) - mu_xy

        num = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        return (num / den).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred01 = (pred + 1.0) * 0.5
        target01 = (target + 1.0) * 0.5
        return 1.0 - self._ssim(pred01, target01)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """Generator 종합 손실 함수.

    L_G = λ_adv · L_adv + λ_L1 · L_L1 + λ_VGG · L_VGG + λ_SSIM · L_SSIM

    Two-stage 학습의 Stage 2 fine-tuning 에서는 ``λ_adv`` 를 매우 작게 (예: 0.01)
    설정하여 GAN 신호가 supervised loss 를 압도하지 않도록 한다.
    """

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_l1: float = 0.7,
        lambda_vgg: float = 0.3,
        lambda_ssim: float = 0.5,
        use_perceptual: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_ssim = lambda_ssim
        self.use_perceptual = use_perceptual

        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual: Optional[PerceptualLoss] = (
            PerceptualLoss() if use_perceptual else None
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        fake_logits: torch.Tensor,
        fake_img: torch.Tensor,
        real_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute generator total loss.

        Parameters
        ----------
        fake_logits : torch.Tensor
            Discriminator(conditional input) output for generated images.
        fake_img : torch.Tensor
            Generator output (range ``[-1, 1]``).
        real_img : torch.Tensor
            Ground-truth (well-lit) image (range ``[-1, 1]``).

        Returns
        -------
        dict[str, Tensor]
            ``total`` 외에 각 항목별 detach 된 모니터링 값 포함.
        """
        target_real = torch.ones_like(fake_logits)
        l_adv = self.adv_criterion(fake_logits, target_real)
        l_l1 = self.l1(fake_img, real_img)
        l_ssim = self.ssim(fake_img, real_img)

        if self.perceptual is not None:
            l_vgg = self.perceptual(fake_img, real_img)
        else:
            l_vgg = torch.zeros((), device=fake_img.device)

        total = (
            self.lambda_adv * l_adv
            + self.lambda_l1 * l_l1
            + self.lambda_vgg * l_vgg
            + self.lambda_ssim * l_ssim
        )

        return {
            "total": total,
            "adv": l_adv.detach(),
            "l1": l_l1.detach(),
            "vgg": l_vgg.detach(),
            "ssim": l_ssim.detach(),
        }


class SupervisedLoss(nn.Module):
    """Stage 1 사전학습용 supervised loss — adversarial term 없음.

    L_G = λ_L1 · L_L1 + λ_VGG · L_VGG + λ_SSIM · L_SSIM

    설계 의도
    ---------
    GAN 직접 학습 시 D 가 G 를 압도하여 PSNR 이 정체되는 현상이 흔하다.
    먼저 G 만 supervised 로 충분히 학습시켜 강한 baseline 을 만들고, 이후
    Stage 2 에서 GAN 으로 텍스처만 미세 조정하는 2-단계 전략의 첫 단계.

    참고: pix2pixHD (Wang et al., 2018), ESRGAN (Wang et al., 2018) 의
          ``PSNR-oriented pre-training`` 도 동일한 동기.
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_vgg: float = 0.5,
        lambda_ssim: float = 1.0,
        use_perceptual: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_ssim = lambda_ssim
        self.use_perceptual = use_perceptual

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual: Optional[PerceptualLoss] = (
            PerceptualLoss() if use_perceptual else None
        )

    def forward(
        self, fake_img: torch.Tensor, real_img: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        l_l1 = self.l1(fake_img, real_img)
        l_ssim = self.ssim(fake_img, real_img)
        if self.perceptual is not None:
            l_vgg = self.perceptual(fake_img, real_img)
        else:
            l_vgg = torch.zeros((), device=fake_img.device)

        total = (
            self.lambda_l1 * l_l1
            + self.lambda_vgg * l_vgg
            + self.lambda_ssim * l_ssim
        )
        return {
            "total": total,
            "l1": l_l1.detach(),
            "vgg": l_vgg.detach(),
            "ssim": l_ssim.detach(),
        }


# ---------------------------------------------------------------------------
# Discriminator loss helper
# ---------------------------------------------------------------------------
class DiscriminatorLoss(nn.Module):
    """PatchGAN Discriminator BCE 손실 + one-sided label smoothing.

    Parameters
    ----------
    real_label : float
        Real 패치의 목표 레이블. 1.0 대신 0.9 등을 쓰면 D 가 logit 을 무한대로
        밀어붙이지 못하게 되어 G 가 학습할 신호가 유지된다 (Salimans 2016,
        "Improved Techniques for Training GANs", §3.4).
    fake_label : float
        Fake 패치 목표 레이블. 일반적으로 0.0 고정.
    """

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.real_label = float(real_label)
        self.fake_label = float(fake_label)

    def forward(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        real_target = torch.full_like(real_logits, self.real_label)
        fake_target = torch.full_like(fake_logits, self.fake_label)
        loss_real = self.bce(real_logits, real_target)
        loss_fake = self.bce(fake_logits, fake_target)
        total = 0.5 * (loss_real + loss_fake)
        return {"total": total, "real": loss_real.detach(), "fake": loss_fake.detach()}
