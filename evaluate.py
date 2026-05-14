"""단독 평가 스크립트.

* 학습된 체크포인트의 Generator 만 로드.
* ``eval15`` 전체에 대해 PSNR / SSIM 평균 계산.
* (옵션) 향상 이미지를 ``results/eval/`` 폴더에 PNG 로 저장.
* CPU/CUDA 추론 latency / FPS 측정.

사용 예::

    python evaluate.py --checkpoint checkpoints/best.pth
    python evaluate.py --checkpoint checkpoints/best.pth --no_save
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

# Windows 콘솔 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
from tqdm import tqdm

from config import EvalConfig
from data import get_eval_loader
from models import LightEnhanceGenerator
from utils import (
    benchmark_inference,
    psnr_metric,
    save_comparison_grid,
    save_tensor_image,
    ssim_metric,
)


HRULE = "=" * 82
SUBRULE = "-" * 82


def load_generator(checkpoint_path: Path, device: str) -> LightEnhanceGenerator:
    """체크포인트에서 Generator 만 로드 (D / optimizer 무시).

    체크포인트에 ``base_filters`` 또는 ``config.base_filters`` 가 저장되어 있으면
    그 값으로 Generator 를 재구성하여 architecture mismatch 를 자동 회피.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # base_filters 자동 감지
    base_filters = 32  # 현재 default
    if isinstance(state, dict):
        if "base_filters" in state:
            base_filters = int(state["base_filters"])
        elif "config" in state and isinstance(state["config"], dict):
            base_filters = int(state["config"].get("base_filters", base_filters))

    G = LightEnhanceGenerator(base_filters=base_filters).to(device)
    if isinstance(state, dict) and "generator" in state:
        G.load_state_dict(state["generator"])
    else:
        G.load_state_dict(state)
    G.eval()
    return G


@torch.no_grad()
def evaluate_dataset(
    G: LightEnhanceGenerator,
    loader,
    device: str,
    save_dir: Path = None,
) -> Dict[str, float]:
    """eval15 loader 전체에 대해 PSNR / SSIM 평균 산출 + 향상 이미지 저장."""
    psnr_sum = ssim_sum = 0.0
    n = 0
    pairs_for_grid = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(loader, desc="Evaluate", ncols=100)
    for idx, (low, high) in enumerate(pbar):
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)

        fake = G(low).clamp(-1.0, 1.0)

        bs = low.size(0)
        psnr_sum += psnr_metric(fake, high) * bs
        ssim_sum += ssim_metric(fake, high) * bs
        n += bs

        if save_dir is not None:
            for b in range(bs):
                stem = f"{idx:03d}_{b:02d}"
                save_tensor_image(low[b].cpu(),  save_dir / f"{stem}_low.png")
                save_tensor_image(fake[b].cpu(), save_dir / f"{stem}_fake.png")
                save_tensor_image(high[b].cpu(), save_dir / f"{stem}_high.png")
                if len(pairs_for_grid) < 9:  # 첫 3 샘플만 비교 그리드
                    pairs_for_grid.extend([
                        low[b].cpu(), fake[b].cpu(), high[b].cpu()
                    ])

        pbar.set_postfix({
            "PSNR(avg)": f"{psnr_sum / n:.2f}",
            "SSIM(avg)": f"{ssim_sum / n:.3f}",
        })

    if save_dir is not None and pairs_for_grid:
        save_comparison_grid(
            pairs_for_grid, save_dir / "comparison_grid.png", ncols=3, pad=4,
        )

    return {
        "psnr": psnr_sum / max(n, 1),
        "ssim": ssim_sum / max(n, 1),
        "n": float(n),
    }


def main() -> int:
    cfg = EvalConfig.parse_args()
    device = cfg.device

    print(HRULE)
    print(" LightEnhanceGAN — 단독 평가")
    print(HRULE)
    print(f"  checkpoint : {cfg.checkpoint}")
    print(f"  data_root  : {cfg.data_root}")
    print(f"  device     : {device}")
    print(SUBRULE)

    ckpt = Path(cfg.checkpoint)
    if not ckpt.is_file():
        print(f"[error] checkpoint not found: {ckpt}")
        return 1
    G = load_generator(ckpt, device)

    loader = get_eval_loader(
        cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
    )
    print(f"  eval pairs : {len(loader.dataset)}")
    print(SUBRULE)

    # ---- 정량 평가 + 이미지 저장 ----
    save_dir = Path(cfg.results_dir) if cfg.save_outputs else None
    t0 = time.perf_counter()
    metrics = evaluate_dataset(G, loader, device=device, save_dir=save_dir)
    eval_sec = time.perf_counter() - t0
    print(SUBRULE)
    print(f"[Metrics over {int(metrics['n'])} pairs]")
    print(f"  PSNR : {metrics['psnr']:.3f} dB")
    print(f"  SSIM : {metrics['ssim']:.4f}")
    print(f"  total eval wall-time : {eval_sec:.2f} s")
    if save_dir is not None:
        print(f"  outputs saved to     : {save_dir.resolve()}")
    print(SUBRULE)

    # ---- FPS 벤치마크 ----
    print(f"[Inference benchmark]")
    bench = benchmark_inference(
        G, (3, cfg.image_size, cfg.image_size),
        device=device, n_warmup=5, n_runs=cfg.bench_runs, batch_size=1,
    )
    print(f"  avg latency : {bench['avg_ms']:.2f} ms / image")
    print(f"  throughput  : {bench['fps']:.1f} FPS  "
          f"(batch=1, runs={int(bench['runs'])}, device={bench['device']})")
    print(HRULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
