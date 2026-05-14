"""모델 분석 유틸리티 — 파라미터 수, MACs/FLOPs, 모델 크기, 추론 벤치마크.

설계 의도
----------
* 외부 의존성(thop, fvcore 등) 없이 PyTorch 만으로 동작하도록 hook-based
  MAC 카운터를 직접 구현. (Jetson 등 임베디드 환경에서 손쉽게 재현 가능)
* 학술 논문 보고용으로 (a) 파라미터 수, (b) MACs (≈FLOPs/2), (c) 디스크
  사이즈(FP32/FP16), (d) 디바이스별 latency/throughput 을 단일 함수로 노출.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Parameter counting
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """전체 / 학습 가능 파라미터 수를 반환.

    Returns
    -------
    (total, trainable) : tuple[int, int]
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# 2. Model size on disk (MB)
# ---------------------------------------------------------------------------
def model_size_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """가중치만 고려한 모델 크기(MB). FP32 = 4 B, FP16/BF16 = 2 B, INT8 = 1 B."""
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)
    total = sum(p.numel() for p in model.parameters())
    total += sum(b.numel() for b in model.buffers())
    return total * bytes_per_param / (1024 ** 2)


# ---------------------------------------------------------------------------
# 3. MAC / FLOP counter (hook-based)
# ---------------------------------------------------------------------------
def compute_macs(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Forward-hook 으로 Conv2d / Linear / DW-Conv 의 MAC 합산.

    Notes
    -----
    * Conv2d MAC = ``K_h · K_w · (C_in / groups) · C_out · H_out · W_out``
    * Depthwise conv 는 groups=in_channels 이므로 위 공식이 자동으로 처리됨.
    * BN/IN/Activation/Pool 은 일반적으로 무시 (전체 대비 ≪1 %).
    * **MAC ≈ FLOPs / 2** (mul + add = 2 ops). 본 함수는 MAC 을 반환.
    """
    macs = [0]

    def conv_hook(module: nn.Conv2d, _inp, out):
        kh, kw = module.kernel_size
        c_in_per_group = module.in_channels // module.groups
        c_out = module.out_channels
        _, _, h, w = out.shape
        macs[0] += kh * kw * c_in_per_group * c_out * h * w

    def linear_hook(module: nn.Linear, _inp, _out):
        macs[0] += module.in_features * module.out_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.zeros(1, *input_shape, device=device)
        model(x)
    if was_training:
        model.train()

    for h in handles:
        h.remove()
    return macs[0]


# ---------------------------------------------------------------------------
# 4. Inference benchmark
# ---------------------------------------------------------------------------
def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
    n_warmup: int = 5,
    n_runs: int = 50,
    batch_size: int = 1,
) -> Dict[str, float]:
    """평균 추론 latency(ms) 와 FPS 측정.

    CUDA 시 ``torch.cuda.synchronize`` 로 정확한 측정.
    임베디드(Jetson) 측정 시 본 함수를 그대로 호출하면 됨.
    """
    model = model.to(device).eval()
    x = torch.randn(batch_size, *input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    avg_ms = elapsed_ms / n_runs
    fps = (1000.0 / avg_ms) * batch_size
    return {
        "avg_ms": avg_ms,
        "fps": fps,
        "runs": float(n_runs),
        "batch_size": float(batch_size),
        "device": device,
    }


# ---------------------------------------------------------------------------
# 5. Helper for pretty-printing large counts
# ---------------------------------------------------------------------------
def format_count(n: float, precision: int = 3) -> str:
    """1234567 → '1.235 M' 형식의 단위 변환 출력."""
    abs_n = abs(n)
    if abs_n >= 1e9:
        return f"{n / 1e9:.{precision}f} G"
    if abs_n >= 1e6:
        return f"{n / 1e6:.{precision}f} M"
    if abs_n >= 1e3:
        return f"{n / 1e3:.{max(precision - 1, 0)}f} K"
    return f"{n:.0f}"
