"""학습 로깅 — 콘솔 / CSV / 샘플 이미지 저장.

* 콘솔: tqdm postfix 로 step 단위 손실 표시, epoch 종료 시 한 줄 요약.
* CSV : ``train_log.csv`` 에 epoch 단위 통계 누적 (그래프용).
* 샘플: ``results/epoch_XXX/`` 폴더에 (low | enhanced | high) 비교 PNG 저장.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Image utilities (torchvision-free 로 grid 합성)
# ---------------------------------------------------------------------------
def _tensor_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """``(3, H, W)`` ∈ ``[-1, 1]`` → ``(H, W, 3)`` uint8."""
    arr = ((t.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return arr.cpu().numpy().transpose(1, 2, 0)


def save_tensor_image(t: torch.Tensor, path: Path) -> None:
    Image.fromarray(_tensor_to_uint8_hwc(t)).save(path)


def save_comparison_grid(
    tensors: List[torch.Tensor], path: Path,
    ncols: int = 3, pad: int = 4, pad_value: int = 255,
) -> None:
    """텐서 리스트를 ncols 컬럼 그리드 PNG 로 저장.

    학습 샘플 시각화에서 일반적으로 ``[low_i, fake_i, high_i]`` 를 한 행으로
    여러 샘플을 쌓아 비교 이미지 한 장을 만든다.
    """
    if not tensors:
        return
    _, H, W = tensors[0].shape
    n = len(tensors)
    rows = (n + ncols - 1) // ncols
    canvas = np.full(
        (rows * H + (rows + 1) * pad, ncols * W + (ncols + 1) * pad, 3),
        pad_value, dtype=np.uint8,
    )
    for k, t in enumerate(tensors):
        r, c = divmod(k, ncols)
        y = pad + r * (H + pad)
        x = pad + c * (W + pad)
        canvas[y:y + H, x:x + W] = _tensor_to_uint8_hwc(t)
    Image.fromarray(canvas).save(path)


# ---------------------------------------------------------------------------
# TrainLogger
# ---------------------------------------------------------------------------
class TrainLogger:
    """학습 로깅 매니저. 콘솔 + CSV + 샘플 이미지를 한곳에서 처리."""

    CSV_FIELDS = [
        "epoch", "step", "phase",
        "d_loss", "g_loss", "g_adv", "g_l1", "g_vgg", "g_ssim",
        "train_psnr", "eval_psnr", "eval_ssim",
        "lr_g", "lr_d", "noise_std", "epoch_sec",
    ]

    def __init__(
        self,
        log_dir: str | Path,
        results_dir: str | Path,
        run_name: Optional[str] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{run_name}" if run_name else ""
        self.csv_path = self.log_dir / f"train_log{suffix}.csv"
        self._init_csv()

    # ------------------------------------------------------------------
    def _init_csv(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.CSV_FIELDS)

    def _append_csv(self, row: Dict[str, Any]) -> None:
        line = [row.get(k, "") for k in self.CSV_FIELDS]
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(line)

    # ==================================================================
    # 콘솔 + CSV
    # ==================================================================
    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_avg: Dict[str, float],
        eval_metrics: Optional[Dict[str, float]],
        best_psnr: float,
        lr_g: float, lr_d: float,
        epoch_sec: float,
    ) -> None:
        """에폭 단위 요약 출력 + CSV 한 줄 추가."""
        ep_str = f"[Epoch {epoch:3d}/{total_epochs}]"
        noise_part = (
            f" | σ_noise: {train_avg['noise_std']:.3f}"
            if "noise_std" in train_avg else ""
        )
        train_str = (
            f"D_loss: {train_avg['d_loss']:.3f}"
            f" | G_loss: {train_avg['g_loss']:.3f}"
            f" | L1: {train_avg['g_l1']:.4f}"
            f" | VGG: {train_avg['g_vgg']:.4f}"
            f" | SSIM(loss): {train_avg['g_ssim']:.4f}"
            f" | PSNR(train): {train_avg['train_psnr']:.2f}"
            f"{noise_part}"
        )
        print(f"{ep_str} {train_str}  ({epoch_sec:.1f}s)")

        eval_str = ""
        if eval_metrics is not None:
            eval_str = (
                f" Eval - PSNR: {eval_metrics['psnr']:.2f}"
                f" | SSIM: {eval_metrics['ssim']:.4f}"
                f" | Best: {best_psnr:.2f}"
            )
            print(f"{ep_str}{eval_str}")

        self._append_csv({
            "epoch": epoch,
            "step": "-",
            "phase": "epoch",
            "d_loss": f"{train_avg['d_loss']:.6f}",
            "g_loss": f"{train_avg['g_loss']:.6f}",
            "g_adv":  f"{train_avg['g_adv']:.6f}",
            "g_l1":   f"{train_avg['g_l1']:.6f}",
            "g_vgg":  f"{train_avg['g_vgg']:.6f}",
            "g_ssim": f"{train_avg['g_ssim']:.6f}",
            "train_psnr": f"{train_avg['train_psnr']:.4f}",
            "eval_psnr":  f"{eval_metrics['psnr']:.4f}" if eval_metrics else "",
            "eval_ssim":  f"{eval_metrics['ssim']:.4f}" if eval_metrics else "",
            "lr_g": f"{lr_g:.2e}",
            "lr_d": f"{lr_d:.2e}",
            "noise_std": f"{train_avg.get('noise_std', 0.0):.4f}",
            "epoch_sec": f"{epoch_sec:.2f}",
        })

    def log_message(self, msg: str) -> None:
        """체크포인트 저장 등 보조 메시지."""
        print(msg)

    # ==================================================================
    # 샘플 이미지 저장
    # ==================================================================
    @torch.no_grad()
    def save_samples(
        self,
        epoch: int,
        generator: torch.nn.Module,
        eval_loader,
        device: str = "cuda",
        n_samples: int = 3,
    ) -> Path:
        """eval 첫 N 샘플의 (low | enhanced | high) 비교 그리드 저장."""
        epoch_dir = self.results_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        was_training = generator.training
        generator.eval()

        tensors: List[torch.Tensor] = []
        saved = 0
        for low, high in eval_loader:
            if saved >= n_samples:
                break
            low_d = low.to(device, non_blocking=True)
            high_d = high.to(device, non_blocking=True)
            fake = generator(low_d).clamp(-1.0, 1.0)

            for b in range(low_d.size(0)):
                if saved >= n_samples:
                    break
                tensors.extend([
                    low_d[b].cpu(),
                    fake[b].cpu(),
                    high_d[b].cpu(),
                ])
                # 개별 이미지도 저장 (논문 figure 용)
                save_tensor_image(low_d[b].cpu(),  epoch_dir / f"sample{saved}_low.png")
                save_tensor_image(fake[b].cpu(),   epoch_dir / f"sample{saved}_fake.png")
                save_tensor_image(high_d[b].cpu(), epoch_dir / f"sample{saved}_high.png")
                saved += 1

        grid_path = epoch_dir / "comparison.png"
        save_comparison_grid(tensors, grid_path, ncols=3, pad=4)

        if was_training:
            generator.train()
        return grid_path

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def now_ts() -> float:
        return time.perf_counter()
