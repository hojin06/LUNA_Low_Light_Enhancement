"""LightEnhanceGAN 학습 메인 스크립트.

전체 흐름
---------
1. ``TrainConfig.parse_args()`` 로 하이퍼파라미터 로드.
2. ``models.LightEnhanceGenerator`` / ``PatchGANDiscriminator`` 생성.
3. Adam(lr=2e-4, β=(0.5, 0.999)) + CosineAnnealingLR scheduler.
4. ``data.get_train_loader / get_eval_loader`` 로 LOL pair 로딩.
5. 매 iteration:
     (a) D step  — real / fake patch logits 에 대한 BCE.
     (b) G step  — Adversarial + λ_L1 L1 + λ_VGG VGG + λ_SSIM SSIM.
   AMP (``torch.amp``) 적용 — RTX 4060 FP16 활용으로 속도/메모리 절감.
6. 매 N epoch:
   - 체크포인트 저장 (``epoch_XXX.pth``, ``last.pth``, eval PSNR 최고 ``best.pth``)
   - eval 샘플 시각화 PNG 저장
7. Ctrl+C / 예외 발생 시에도 ``last.pth`` 가 보존되도록 ``try/finally``.

설계 노트
---------
* **fake 한 번 계산, 두 번 사용**: D step 에서는 ``fake.detach()`` 로 G 의 그
  래디언트 흐름 차단, G step 에서는 동일 ``fake`` 텐서 재사용 → 계산 1 회.
* **requires_grad 토글**: G step 동안 D 의 ``requires_grad=False`` 로 D 의
  파라미터에 불필요한 그래디언트 누적 방지 (속도/메모리).
* **OOM fallback**: 사용자 ``batch_size`` 로 첫 step 실패 시 ``//2`` 로 재시도.

GAN 안정화 (D 과수렴 방지) — 모두 ``TrainConfig`` 로 조절 가능:
* **LR 분리** (`lr_d` ≪ `lr_g`): D 가 더 천천히 학습되어 G 가 따라잡을 여지.
* **D 업데이트 빈도** (`d_update_freq`): G N step 당 D 1 step.
* **One-sided label smoothing** (`label_smoothing_real`): D 의 real 목표를
  0.9 로 두어 logit polarization 억제. Salimans et al., 2016.
* **Instance noise**: D 의 real/fake 입력에 std σ_t 의 Gaussian noise 주입.
  σ_t 는 epoch 진행에 따라 선형 감쇠. Sønderby et al., 2017.
* **Spectral normalization**: D 의 Conv 에 SN 적용 (D 의 Lipschitz≤1).
  Miyato et al., 2018.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Windows 콘솔(cp949) 환경에서도 한글/유니코드 출력 안전.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import TrainConfig
from data import get_eval_loader, get_train_loader
from models import (
    CombinedLoss,
    DiscriminatorLoss,
    LightEnhanceGenerator,
    PatchGANDiscriminator,
)
from utils import TrainLogger, evaluate, psnr_metric


# ===========================================================================
# Reproducibility / IO helpers
# ===========================================================================
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    epoch: int,
    G: nn.Module, D: nn.Module,
    opt_g, opt_d,
    sch_g, sch_d,
    sc_g: GradScaler, sc_d: GradScaler,
    best_psnr: float,
    cfg: TrainConfig,
) -> None:
    """학습 재개에 필요한 전체 state 직렬화."""
    state = {
        "epoch": epoch,
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "sch_g": sch_g.state_dict(),
        "sch_d": sch_d.state_dict(),
        "sc_g": sc_g.state_dict(),
        "sc_d": sc_d.state_dict(),
        "best_psnr": best_psnr,
        "config": cfg.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def instance_noise_std_for_epoch(
    epoch: int, init_std: float, decay_epochs: int
) -> float:
    """Epoch 별 D 입력 noise σ. 1 epoch 에서 init_std, decay_epochs 이후 0.

    선형 감쇠 스케줄 (Sønderby et al., 2017):
        σ(t) = init_std · max(0, 1 - (t-1)/decay_epochs)
    """
    if init_std <= 0.0 or decay_epochs <= 0:
        return 0.0
    progress = min(1.0, max(0.0, (epoch - 1) / decay_epochs))
    return float(init_std * (1.0 - progress))


def _add_gaussian(x: torch.Tensor, std: float) -> torch.Tensor:
    """std > 0 일 때만 fresh Gaussian noise 를 더해 반환 (in-place 아님)."""
    if std <= 0.0:
        return x
    return x + torch.randn_like(x) * std


def _d_forward_fp32(D: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Discriminator forward 를 **항상 FP32 로** 실행.

    설계 의도
    ---------
    Spectral Norm 의 power iteration 은 내부 buffer ``_u``, ``_v`` 를 매 forward
    마다 in-place 업데이트한다. AMP autocast 아래에서 FP16 으로 한 번이라도
    overflow 가 발생하면 이 buffer 가 inf/NaN 으로 굳어 영구적으로 D 가 망가진다.
    또한 lr_d 가 매우 작은 경우 (예: 1e-5) FP16 의 표현 한계가 그래디언트 정밀
    도를 훼손한다. D 만 FP32 로 분리하면 G 의 큰 conv 는 AMP 의 이득을 그대로
    누리면서 D 학습은 안정적으로 진행된다.
    """
    if x.dtype != torch.float32:
        x = x.float()
    with autocast(device_type="cuda", enabled=False):
        return D(x)


def load_checkpoint(
    path: Path,
    G: nn.Module, D: nn.Module,
    opt_g=None, opt_d=None,
    sch_g=None, sch_d=None,
    sc_g: Optional[GradScaler] = None, sc_d: Optional[GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    G.load_state_dict(state["generator"])
    D.load_state_dict(state["discriminator"])
    if opt_g is not None and "opt_g" in state: opt_g.load_state_dict(state["opt_g"])
    if opt_d is not None and "opt_d" in state: opt_d.load_state_dict(state["opt_d"])
    if sch_g is not None and "sch_g" in state: sch_g.load_state_dict(state["sch_g"])
    if sch_d is not None and "sch_d" in state: sch_d.load_state_dict(state["sch_d"])
    if sc_g is not None and "sc_g" in state: sc_g.load_state_dict(state["sc_g"])
    if sc_d is not None and "sc_d" in state: sc_d.load_state_dict(state["sc_d"])
    return state


# ===========================================================================
# One epoch
# ===========================================================================
def train_one_epoch(
    epoch: int,
    cfg: TrainConfig,
    G: nn.Module, D: nn.Module,
    opt_g, opt_d,
    sc_g: GradScaler, sc_d: GradScaler,
    combined_loss: CombinedLoss,
    d_loss_fn: DiscriminatorLoss,
    train_loader,
    device: str,
) -> Dict[str, float]:
    """한 epoch 학습. 손실/메트릭 누적 평균 반환.

    안정화 적용 사항
    ----------------
    * ``cfg.d_update_freq`` 마다 D step 1 회 (G step 은 매 iter).
    * D 입력 (real / fake) 에 epoch 종속 σ 의 Gaussian noise 주입.
    * AMP (FP16) 적용 + per-step requires_grad 토글로 backward 비용 절감.
    """
    G.train()
    D.train()

    # epoch 별 instance noise σ (real / fake 양쪽 D 입력에 동일 σ 적용)
    noise_std = instance_noise_std_for_epoch(
        epoch, cfg.instance_noise_std, cfg.instance_noise_decay_epochs,
    )

    sums = {
        "d_loss": 0.0, "g_loss": 0.0,
        "g_adv": 0.0, "g_l1": 0.0, "g_vgg": 0.0, "g_ssim": 0.0,
        "train_psnr": 0.0,
    }
    n_samples = 0       # G step 횟수 (= 전체 iter 수) 가중 합용
    d_sample_total = 0  # D step 이 실제로 돌아간 샘플 수
    use_amp = cfg.amp and device == "cuda"

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:3d}/{cfg.num_epochs}",
        ncols=120, leave=False, dynamic_ncols=False,
    )

    for step, (low, high) in enumerate(pbar):
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        bs = low.size(0)

        # =====================================================
        # 공통: fake 한 번만 계산 (D / G 양쪽에서 재사용)
        # =====================================================
        with autocast(device_type="cuda", enabled=use_amp):
            fake = G(low)

        # =====================================================
        # (1) Discriminator step  (매 d_update_freq 회만)
        # =====================================================
        do_d_step = (step % max(cfg.d_update_freq, 1)) == 0
        last_d_loss: float = float("nan")
        if do_d_step:
            for p in D.parameters():
                p.requires_grad = True
            opt_d.zero_grad(set_to_none=True)

            # D 입력은 FP32 로 준비 (instance noise 도 FP32 도메인)
            real_pair = _add_gaussian(
                torch.cat([low, high], dim=1).float(), noise_std)
            fake_pair = _add_gaussian(
                torch.cat([low, fake.detach()], dim=1).float(), noise_std)

            d_real_logits = _d_forward_fp32(D, real_pair)
            d_fake_logits = _d_forward_fp32(D, fake_pair)
            d_losses = d_loss_fn(d_real_logits, d_fake_logits)
            loss_d = d_losses["total"]

            sc_d.scale(loss_d).backward()
            sc_d.step(opt_d)
            sc_d.update()

            d_val = float(loss_d.detach())
            sums["d_loss"]   += d_val * bs
            d_sample_total   += bs
            last_d_loss       = d_val

        # =====================================================
        # (2) Generator step (D frozen, 매 iter)
        # =====================================================
        for p in D.parameters():
            p.requires_grad = False
        opt_g.zero_grad(set_to_none=True)

        # D 는 FP32 로 평가하되 (SN 안정성), 나머지 G 손실은 AMP 활용.
        g_d_input = _add_gaussian(
            torch.cat([low, fake], dim=1).float(), noise_std)
        d_fake_for_g = _d_forward_fp32(D, g_d_input)
        with autocast(device_type="cuda", enabled=use_amp):
            g_losses = combined_loss(d_fake_for_g, fake, high)
            loss_g = g_losses["total"]

        sc_g.scale(loss_g).backward()
        sc_g.step(opt_g)
        sc_g.update()

        # =====================================================
        # (3) 통계
        # =====================================================
        with torch.no_grad():
            train_psnr = psnr_metric(fake.detach(), high)

        sums["g_loss"]     += float(loss_g.detach()) * bs
        sums["g_adv"]      += float(g_losses["adv"]) * bs
        sums["g_l1"]       += float(g_losses["l1"])  * bs
        sums["g_vgg"]      += float(g_losses["vgg"]) * bs
        sums["g_ssim"]     += float(g_losses["ssim"]) * bs
        sums["train_psnr"] += train_psnr * bs
        n_samples += bs

        pbar.set_postfix({
            "D":     f"{last_d_loss:.3f}" if do_d_step else "  -  ",
            "G":     f"{float(loss_g):.3f}",
            "L1":    f"{float(g_losses['l1']):.3f}",
            "PSNR":  f"{train_psnr:.1f}",
            "noise": f"{noise_std:.3f}",
        })

    pbar.close()

    # 평균 (D 통계는 D 가 실제 돈 iter 만으로 평균)
    avg = {k: v / max(n_samples, 1) for k, v in sums.items() if k != "d_loss"}
    avg["d_loss"] = sums["d_loss"] / max(d_sample_total, 1)
    avg["noise_std"] = noise_std
    return avg


# ===========================================================================
# Build & run
# ===========================================================================
def _build(cfg: TrainConfig, device: str):
    """모델/옵티마이저/스케줄러/스케일러/손실/로더를 모두 구성."""
    G = LightEnhanceGenerator(base_filters=cfg.base_filters).to(device)
    D = PatchGANDiscriminator(
        use_spectral_norm=cfg.use_spectral_norm,
    ).to(device)

    opt_g = Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    sch_g = CosineAnnealingLR(opt_g, T_max=cfg.num_epochs,
                              eta_min=cfg.lr_g * cfg.eta_min_ratio)
    sch_d = CosineAnnealingLR(opt_d, T_max=cfg.num_epochs,
                              eta_min=cfg.lr_d * cfg.eta_min_ratio)

    use_amp = cfg.amp and device == "cuda"
    sc_g = GradScaler(device="cuda", enabled=use_amp)
    sc_d = GradScaler(device="cuda", enabled=use_amp)

    combined_loss = CombinedLoss(
        lambda_adv=cfg.lambda_adv,
        lambda_l1=cfg.lambda_l1,
        lambda_vgg=cfg.lambda_perceptual,
        lambda_ssim=cfg.lambda_ssim,
    ).to(device)
    d_loss_fn = DiscriminatorLoss(real_label=cfg.label_smoothing_real).to(device)

    train_loader = get_train_loader(
        cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        seed=cfg.seed,
        full_resize=cfg.full_resize,
    )
    eval_loader = get_eval_loader(
        cfg.data_root,
        batch_size=1,
        num_workers=min(cfg.num_workers, 2),
        image_size=cfg.image_size,
    )

    return (G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
            combined_loss, d_loss_fn, train_loader, eval_loader)


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = cfg.device

    # Spectral Norm + AMP 호환성 문제:
    # SN 의 power-iteration 결과 sigma 가 backward 시 GradScaler 의 loss-scaling
    # 과 상호작용하여 그래디언트가 inf 로 폭주하고, 이로 인해 D 의 weight_orig
    # 가 NaN 으로 굳는 현상이 재현된다 (PyTorch 2.5 / parametrizations API).
    # 단순 D-forward-FP32 만으로는 해결되지 않아, SN 사용 시 AMP 를 강제 비활성.
    if cfg.use_spectral_norm and cfg.amp and device == "cuda":
        print("[train] WARN: spectral_norm + AMP 조합은 NaN 을 유발하므로 "
              "AMP 를 자동으로 비활성합니다. (--no_spectral_norm 으로 AMP 유지 가능)")
        cfg.amp = False

    print(f"[train] device={device}, AMP={'on' if cfg.amp and device == 'cuda' else 'off'}, "
          f"batch_size={cfg.batch_size}")
    print(f"[train] lr_g={cfg.lr_g:.2e}, lr_d={cfg.lr_d:.2e}  "
          f"(ratio={cfg.lr_g / max(cfg.lr_d, 1e-12):.1f}×)")
    print(f"[train] GAN 안정화 - d_update_freq={cfg.d_update_freq}, "
          f"label_smoothing_real={cfg.label_smoothing_real}, "
          f"instance_noise σ={cfg.instance_noise_std} "
          f"(decay over {cfg.instance_noise_decay_epochs} ep), "
          f"spectral_norm={'on' if cfg.use_spectral_norm else 'off'}")

    (G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
     combined_loss, d_loss_fn, train_loader, eval_loader) = _build(cfg, device)

    # ---- 재개 ----
    start_epoch = 1
    best_psnr = -float("inf")
    if cfg.resume:
        ckpt_path = Path(cfg.resume)
        if ckpt_path.is_file():
            state = load_checkpoint(
                ckpt_path, G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                map_location=device,
            )
            start_epoch = int(state.get("epoch", 0)) + 1
            best_psnr = float(state.get("best_psnr", -float("inf")))
            print(f"[resume] {ckpt_path} — start_epoch={start_epoch}, "
                  f"best_psnr={best_psnr:.2f}")
        else:
            print(f"[resume] WARN: checkpoint not found: {ckpt_path}  (처음부터 시작)")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(cfg.log_dir, cfg.results_dir)

    # ====================================================================
    # Main training loop
    # ====================================================================
    last_completed = start_epoch - 1
    try:
        for epoch in range(start_epoch, cfg.num_epochs + 1):
            t0 = time.perf_counter()
            train_avg = train_one_epoch(
                epoch=epoch, cfg=cfg,
                G=G, D=D, opt_g=opt_g, opt_d=opt_d, sc_g=sc_g, sc_d=sc_d,
                combined_loss=combined_loss, d_loss_fn=d_loss_fn,
                train_loader=train_loader, device=device,
            )
            sch_g.step()
            sch_d.step()
            epoch_sec = time.perf_counter() - t0

            # ---- eval ----
            eval_metrics: Optional[Dict[str, float]] = None
            if epoch % cfg.eval_every == 0:
                eval_metrics = evaluate(G, eval_loader, device=device)
                if eval_metrics["psnr"] > best_psnr:
                    best_psnr = eval_metrics["psnr"]
                    save_checkpoint(save_dir / "best.pth",
                                    epoch, G, D, opt_g, opt_d, sch_g, sch_d,
                                    sc_g, sc_d, best_psnr, cfg)
                    logger.log_message(
                        f"  -> new best PSNR={best_psnr:.2f}  (saved best.pth)"
                    )

            logger.log_epoch(
                epoch=epoch, total_epochs=cfg.num_epochs,
                train_avg=train_avg, eval_metrics=eval_metrics,
                best_psnr=best_psnr,
                lr_g=opt_g.param_groups[0]["lr"],
                lr_d=opt_d.param_groups[0]["lr"],
                epoch_sec=epoch_sec,
            )

            # ---- 체크포인트 ----
            if epoch % cfg.save_every == 0:
                ckpt = save_dir / f"epoch_{epoch:03d}.pth"
                save_checkpoint(ckpt, epoch, G, D, opt_g, opt_d, sch_g, sch_d,
                                sc_g, sc_d, best_psnr, cfg)
                logger.log_message(f"  checkpoint: {ckpt}")

            # ---- 샘플 시각화 ----
            if epoch % cfg.sample_every == 0:
                grid_path = logger.save_samples(epoch, G, eval_loader,
                                                device=device, n_samples=3)
                logger.log_message(f"  samples   : {grid_path}")

            last_completed = epoch

    except KeyboardInterrupt:
        print("\n[train] Ctrl+C — saving last checkpoint and exiting...")
    except Exception as exc:
        print(f"\n[train] Exception during training: {exc}")
        raise
    finally:
        # 마지막 상태를 last.pth 로 항상 저장 (재개용)
        last_path = save_dir / "last.pth"
        save_checkpoint(last_path, last_completed,
                        G, D, opt_g, opt_d, sch_g, sch_d, sc_g, sc_d,
                        best_psnr, cfg)
        print(f"[train] last checkpoint -> {last_path}  "
              f"(epoch={last_completed}, best_psnr={best_psnr:.2f})")


# ===========================================================================
# Entry point with OOM fallback
# ===========================================================================
def main() -> int:
    cfg = TrainConfig.parse_args()

    # batch_size 후보: 사용자 지정 → //2 → //4 → 1
    candidates = [cfg.batch_size]
    if cfg.auto_oom_fallback and cfg.batch_size > 1:
        b = cfg.batch_size
        while b > 1:
            b = max(b // 2, 1)
            candidates.append(b)

    last_err: Optional[BaseException] = None
    for bs in candidates:
        cfg.batch_size = bs
        try:
            run_training(cfg)
            return 0
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            last_err = e
            print(f"[OOM] batch_size={bs} 실패. 메모리 비우고 더 작은 배치로 재시도.")
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            # 일부 torch 버전은 OOM 을 일반 RuntimeError 로 던짐
            if "out of memory" in str(e).lower():
                last_err = e
                print(f"[OOM] batch_size={bs} 실패: {e}\n  더 작은 배치로 재시도.")
                torch.cuda.empty_cache()
                continue
            raise

    print(f"[train] 모든 batch_size 후보 실패: {candidates}.  마지막 에러: {last_err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
