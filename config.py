"""학습/평가 하이퍼파라미터 정의 + argparse 헬퍼.

설계
----
* 모든 default 값을 ``@dataclass`` 한 곳에서 관리 → argparse 는 그 default 를
  그대로 노출. 코드와 CLI 가 어긋날 일이 없다.
* ``TrainConfig.parse_args()`` 한 번 호출로 dataclass 인스턴스를 얻는다.
* checkpoint 에 ``config.to_dict()`` 로 직렬화해 함께 저장 → 실험 재현성 확보.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """학습 전 과정에서 참조되는 설정 묶음."""

    # ---- 데이터 ----
    data_root: str = "../DataSet/LOLdataset"
    image_size: int = 256
    num_workers: int = 4
    full_resize: bool = False     # True 면 random crop 없이 전체 리사이즈

    # ---- 모델 ----
    base_filters: int = 32        # Generator 의 base channel (24→32, capacity↑)

    # ---- 최적화 ----
    batch_size: int = 8
    lr_g: float = 2e-4
    lr_d: float = 1e-5           # G 대비 20× 느리게. D 과수렴 방지.
    beta1: float = 0.5           # GAN 표준 (Radford et al., DCGAN)
    beta2: float = 0.999
    num_epochs: int = 200
    eta_min_ratio: float = 0.01  # CosineAnnealing 의 최소 lr 비율

    # ---- 손실 가중치 ----
    lambda_adv: float = 1.0       # adversarial loss 가중치 (Stage 2 에선 0.01 권장)
    lambda_l1: float = 0.7
    lambda_perceptual: float = 0.3
    lambda_ssim: float = 0.5

    # ---- GAN 안정화 ----
    d_update_freq: int = 2                # G step n 회당 D step 1 회 (G:D = 2:1)
    label_smoothing_real: float = 0.9     # one-sided label smoothing (Salimans 2016)
    instance_noise_std: float = 0.1       # D 입력 Gaussian noise 초기 표준편차
    instance_noise_decay_epochs: int = 50 # noise std → 0 까지 선형 감쇠 epoch
    use_spectral_norm: bool = True        # Discriminator spectral norm (Miyato 2018)

    # ---- I/O ----
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: str = "./results"
    resume: Optional[str] = None

    # ---- 주기 ----
    save_every: int = 10          # 에폭마다 체크포인트 저장 간격
    sample_every: int = 5         # 샘플 시각화 저장 간격
    eval_every: int = 1           # 평가 주기

    # ---- 런타임 ----
    amp: bool = True              # Mixed precision (FP16)
    device: str = "cuda"          # "cuda" | "cpu"
    seed: int = 42
    auto_oom_fallback: bool = True  # OOM 시 batch_size 절반으로 재시도

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------
    @classmethod
    def parse_args(cls, argv: Optional[list[str]] = None) -> "TrainConfig":
        """CLI 인자 파싱 → TrainConfig 인스턴스."""
        d = cls()  # default 보존용
        p = argparse.ArgumentParser(description="LightEnhanceGAN 학습")

        # 데이터
        p.add_argument("--data_root", type=str, default=d.data_root)
        p.add_argument("--image_size", type=int, default=d.image_size)
        p.add_argument("--num_workers", type=int, default=d.num_workers)
        p.add_argument("--full_resize", action="store_true",
                       help="random crop 없이 전체 이미지를 image_size 로 리사이즈")
        # 모델
        p.add_argument("--base_filters", type=int, default=d.base_filters,
                       help="Generator 의 base channel 수 (capacity 조절).")
        # 최적화
        p.add_argument("--batch_size", type=int, default=d.batch_size)
        p.add_argument("--lr_g", type=float, default=d.lr_g)
        p.add_argument("--lr_d", type=float, default=d.lr_d)
        p.add_argument("--beta1", type=float, default=d.beta1)
        p.add_argument("--beta2", type=float, default=d.beta2)
        p.add_argument("--num_epochs", type=int, default=d.num_epochs)
        p.add_argument("--eta_min_ratio", type=float, default=d.eta_min_ratio)
        # 손실 가중치
        p.add_argument("--lambda_adv", type=float, default=d.lambda_adv,
                       help="adversarial loss 가중치 (Stage 2 fine-tune 시 0.01 권장)")
        p.add_argument("--lambda_l1", type=float, default=d.lambda_l1)
        p.add_argument("--lambda_perceptual", type=float, default=d.lambda_perceptual)
        p.add_argument("--lambda_ssim", type=float, default=d.lambda_ssim)
        # GAN 안정화
        p.add_argument("--d_update_freq", type=int, default=d.d_update_freq,
                       help="G step n 번당 D step 1 번 (기본 2)")
        p.add_argument("--label_smoothing_real", type=float,
                       default=d.label_smoothing_real,
                       help="D 학습 시 real 레이블 값 (기본 0.9)")
        p.add_argument("--instance_noise_std", type=float,
                       default=d.instance_noise_std,
                       help="D 입력 초기 noise σ. 0 이면 비활성.")
        p.add_argument("--instance_noise_decay_epochs", type=int,
                       default=d.instance_noise_decay_epochs,
                       help="noise σ → 0 까지의 선형 감쇠 epoch 수.")
        p.add_argument("--no_spectral_norm", action="store_true",
                       help="Discriminator spectral norm 비활성")
        # I/O
        p.add_argument("--save_dir", type=str, default=d.save_dir)
        p.add_argument("--log_dir", type=str, default=d.log_dir)
        p.add_argument("--results_dir", type=str, default=d.results_dir)
        p.add_argument("--resume", type=str, default=d.resume,
                       help="checkpoint 경로 (없으면 처음부터 학습)")
        # 주기
        p.add_argument("--save_every", type=int, default=d.save_every)
        p.add_argument("--sample_every", type=int, default=d.sample_every)
        p.add_argument("--eval_every", type=int, default=d.eval_every)
        # 런타임
        p.add_argument("--no_amp", action="store_true",
                       help="Mixed Precision (AMP) 비활성")
        p.add_argument("--device", type=str, default=None,
                       help="cuda|cpu, 미지정 시 자동 감지")
        p.add_argument("--seed", type=int, default=d.seed)
        p.add_argument("--no_oom_fallback", action="store_true",
                       help="OOM 시 batch_size 자동 축소 비활성")

        a = p.parse_args(argv)

        # device auto-detect: import 순환을 피하려고 여기서만 torch 호출
        if a.device is None:
            import torch
            a.device = "cuda" if torch.cuda.is_available() else "cpu"

        return cls(
            data_root=a.data_root,
            image_size=a.image_size,
            num_workers=a.num_workers,
            full_resize=a.full_resize,
            base_filters=a.base_filters,
            batch_size=a.batch_size,
            lr_g=a.lr_g,
            lr_d=a.lr_d,
            beta1=a.beta1,
            beta2=a.beta2,
            num_epochs=a.num_epochs,
            eta_min_ratio=a.eta_min_ratio,
            lambda_adv=a.lambda_adv,
            lambda_l1=a.lambda_l1,
            lambda_perceptual=a.lambda_perceptual,
            lambda_ssim=a.lambda_ssim,
            d_update_freq=a.d_update_freq,
            label_smoothing_real=a.label_smoothing_real,
            instance_noise_std=a.instance_noise_std,
            instance_noise_decay_epochs=a.instance_noise_decay_epochs,
            use_spectral_norm=not a.no_spectral_norm,
            save_dir=a.save_dir,
            log_dir=a.log_dir,
            results_dir=a.results_dir,
            resume=a.resume,
            save_every=a.save_every,
            sample_every=a.sample_every,
            eval_every=a.eval_every,
            amp=not a.no_amp,
            device=a.device,
            seed=a.seed,
            auto_oom_fallback=not a.no_oom_fallback,
        )


@dataclass
class EvalConfig:
    """단독 평가 스크립트 (``evaluate.py``) 용 설정."""

    data_root: str = "../DataSet/LOLdataset"
    checkpoint: str = "./checkpoints/best.pth"
    results_dir: str = "./results/eval"
    image_size: int = 256
    batch_size: int = 1
    num_workers: int = 2
    device: str = "cuda"
    save_outputs: bool = True
    bench_runs: int = 50           # FPS 측정 반복 횟수

    @classmethod
    def parse_args(cls, argv: Optional[list[str]] = None) -> "EvalConfig":
        d = cls()
        p = argparse.ArgumentParser(description="LightEnhanceGAN 평가")
        p.add_argument("--data_root", type=str, default=d.data_root)
        p.add_argument("--checkpoint", type=str, default=d.checkpoint)
        p.add_argument("--results_dir", type=str, default=d.results_dir)
        p.add_argument("--image_size", type=int, default=d.image_size)
        p.add_argument("--batch_size", type=int, default=d.batch_size)
        p.add_argument("--num_workers", type=int, default=d.num_workers)
        p.add_argument("--device", type=str, default=None)
        p.add_argument("--no_save", action="store_true",
                       help="향상 이미지 저장 비활성")
        p.add_argument("--bench_runs", type=int, default=d.bench_runs)
        a = p.parse_args(argv)

        if a.device is None:
            import torch
            a.device = "cuda" if torch.cuda.is_available() else "cpu"

        return cls(
            data_root=a.data_root,
            checkpoint=a.checkpoint,
            results_dir=a.results_dir,
            image_size=a.image_size,
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            device=a.device,
            save_outputs=not a.no_save,
            bench_runs=a.bench_runs,
        )
