"""LightEnhanceGAN 아키텍처 검증/분석 스크립트.

실행 시 출력 항목
------------------
1. Generator / Discriminator 파라미터 수 (total / trainable)
2. MACs (256×256 입력 기준), FLOPs ≈ 2·MACs
3. 모델 크기 (FP32 / FP16)
4. 기존 저조도 향상 모델과의 비교표
5. Forward pass shape / 출력 범위 sanity check
6. CombinedLoss 동작 확인
7. 추론 벤치마크 (CPU 또는 CUDA, 자동 선택)

사용법:
    python test_architecture.py
"""
from __future__ import annotations

import sys
from typing import List, Tuple

# Windows 콘솔(cp949)에서도 한글/유니코드(em-dash 등) 출력이 깨지지 않도록 UTF-8 강제.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    
import torch

from models import (
    CombinedLoss,
    LightEnhanceGenerator,
    PatchGANDiscriminator,
)
from utils import (
    benchmark_inference,
    compute_macs,
    count_parameters,
    format_count,
    model_size_mb,
)


# ---------------------------------------------------------------------------
HRULE = "=" * 82
SUBRULE = "-" * 82


def _print_header(title: str) -> None:
    print(HRULE)
    print(f" {title}")
    print(HRULE)


# ---------------------------------------------------------------------------
# 1. Module-level diagnostics
# ---------------------------------------------------------------------------
def analyze_module(name: str, model: torch.nn.Module, input_shape: Tuple[int, ...]) -> dict:
    total, trainable = count_parameters(model)
    macs = compute_macs(model, input_shape)
    size_fp32 = model_size_mb(model, torch.float32)
    size_fp16 = model_size_mb(model, torch.float16)

    print(f"[{name}]")
    print(f"  Parameters    : total = {format_count(total)} ({total:,})"
          f"  |  trainable = {format_count(trainable)}")
    print(f"  Model size    : FP32 = {size_fp32:.3f} MB"
          f"  |  FP16 = {size_fp16:.3f} MB")
    print(f"  MACs (1×{input_shape[0]}×{input_shape[1]}×{input_shape[2]})"
          f"  : {format_count(macs)}   (FLOPs ≈ {format_count(2 * macs)})")
    print(SUBRULE)
    return {
        "params": total, "trainable": trainable,
        "macs": macs, "fp32_mb": size_fp32, "fp16_mb": size_fp16,
    }


# ---------------------------------------------------------------------------
# 2. Forward pass sanity check
# ---------------------------------------------------------------------------
def forward_sanity(G: torch.nn.Module, D: torch.nn.Module) -> None:
    _print_header("Forward pass sanity check")
    x_low = torch.randn(2, 3, 256, 256).clamp(-1.0, 1.0)
    with torch.no_grad():
        y_hat = G(x_low)
        d_input = torch.cat([x_low, y_hat], dim=1)
        d_out = D(d_input)

    print(f"  Generator     : in {tuple(x_low.shape)}  ->  out {tuple(y_hat.shape)}")
    print(f"  Output range  : min = {y_hat.min().item():+.4f}"
          f"  |  max = {y_hat.max().item():+.4f}"
          f"  |  mean = {y_hat.mean().item():+.4f}")
    in_range = (y_hat.min() >= -1.0 - 1e-6) and (y_hat.max() <= 1.0 + 1e-6)
    print(f"  Tanh in [-1,1]: {'OK' if in_range else 'FAIL'}")
    print(f"  Discriminator : in {tuple(d_input.shape)}  ->  out {tuple(d_out.shape)}")
    print(f"  Expected D out: (B, 1, 16, 16) — PatchGAN logits")
    print(SUBRULE)


# ---------------------------------------------------------------------------
# 3. Combined loss check
# ---------------------------------------------------------------------------
def loss_sanity(G: torch.nn.Module, D: torch.nn.Module) -> None:
    _print_header("CombinedLoss sanity check")
    try:
        loss_fn = CombinedLoss(lambda_l1=0.7, lambda_vgg=0.3, lambda_ssim=0.5)
    except Exception as exc:  # pragma: no cover
        print(f"  (CombinedLoss init failed: {exc})")
        print(SUBRULE)
        return

    x_low = torch.randn(2, 3, 256, 256).clamp(-1.0, 1.0)
    real = torch.randn(2, 3, 256, 256).clamp(-1.0, 1.0)
    with torch.no_grad():
        fake = G(x_low)
        d_fake = D(torch.cat([x_low, fake], dim=1))
    losses = loss_fn(d_fake, fake, real)
    for k, v in losses.items():
        print(f"  {k:<6s}: {float(v):.4f}")
    print(SUBRULE)


# ---------------------------------------------------------------------------
# 4. Comparison table with prior work
# ---------------------------------------------------------------------------
def comparison_table(g_params: int, g_macs: int) -> None:
    _print_header("기존 저조도 향상 모델과의 비교 (256×256 기준)")
    # 비고: Params/MACs 는 각 논문 또는 공식 코드 기준 근사치.
    #        실측치와 차이가 있을 수 있으므로 학술 논문 게재 시 재측정 권장.
    rows: List[Tuple[str, str, str, str, str]] = [
        # (Method,           Params,    MACs,      Sup.,        Notes)
        ("Zero-DCE   (CVPR'20)",       "79.4 K",   "~5.21 G",   "Unsup.",  "Curve estimation"),
        ("Zero-DCE++ (TPAMI'21)",      "10.6 K",   "~0.11 G",   "Unsup.",  "DSConv, ultra-light"),
        ("SCI        (CVPR'22)",       "10.6 K",   "~0.18 G",   "Unsup.",  "Self-calibrated"),
        ("RetinexNet (BMVC'18)",       "555 K",    "~587 G",    "Sup.",    "Decom + Enhance"),
        ("EnlightenGAN (TIP'21)",      "8.64 M",   "~273 G",    "Unpair.", "Global-local D + attn."),
        ("FUnIE-GAN  (RA-L'20)",       "7.02 M",   "~10.2 G",   "Paired",  "Underwater U-Net+PatchGAN"),
        ("Ours  (LightEnhanceGAN)",
         format_count(g_params),
         f"{g_macs / 1e9:.3f} G",
         "Paired",
         "DSConv U-Net + CA+SA"),
    ]
    hdr = f"  {'Model':<26} {'Params':>10} {'MACs':>10}  {'Sup.':<8} {'Notes'}"
    print(hdr)
    print(f"  {'-'*26} {'-'*10} {'-'*10}  {'-'*8} {'-'*30}")
    for name, p, m, s, n in rows:
        marker = "  <—" if name.startswith("Ours") else ""
        print(f"  {name:<26} {p:>10} {m:>10}  {s:<8} {n}{marker}")
    print(SUBRULE)
    print("  * 수치는 공개 논문/저자 코드의 근사치이며, 실험 환경에 따라 변동.")
    print(SUBRULE)


# ---------------------------------------------------------------------------
# 5. Inference benchmark
# ---------------------------------------------------------------------------
def benchmark_section(G: torch.nn.Module) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _print_header(f"Inference benchmark  (device = {device})")

    bench = benchmark_inference(
        G, (3, 256, 256),
        device=device, n_warmup=5, n_runs=30, batch_size=1,
    )
    print(f"  avg latency : {bench['avg_ms']:>7.2f} ms / image")
    print(f"  throughput  : {bench['fps']:>7.1f} FPS"
          f"   (batch={int(bench['batch_size'])}, runs={int(bench['runs'])})")
    print(f"  target      : >= 30 FPS on Jetson Orin Nano (15 W TDP)")
    print(SUBRULE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="아키텍처 분석")
    p.add_argument("--base_filters", type=int, default=32,
                   help="Generator base channel (24 / 32 등 capacity ablation 용)")
    p.add_argument("--d_base_filters", type=int, default=32,
                   help="Discriminator base channel")
    p.add_argument("--use_spectral_norm", action="store_true",
                   help="Discriminator 에 spectral norm 적용 (param 수 동일)")
    args = p.parse_args()

    torch.manual_seed(0)

    _print_header("LightEnhanceGAN — 아키텍처 분석 리포트")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  CUDA             : {'available' if torch.cuda.is_available() else 'unavailable'}")
    print(f"  G.base_filters   : {args.base_filters}")
    print(f"  D.base_filters   : {args.d_base_filters}")
    print(f"  D.spectral_norm  : {'on' if args.use_spectral_norm else 'off'}")
    print(SUBRULE)

    G = LightEnhanceGenerator(in_channels=3, out_channels=3, base_filters=args.base_filters)
    D = PatchGANDiscriminator(
        in_channels=6, base_filters=args.d_base_filters,
        use_spectral_norm=args.use_spectral_norm,
    )

    g_stats = analyze_module("Generator", G, input_shape=(3, 256, 256))
    _ = analyze_module("Discriminator", D, input_shape=(6, 256, 256))

    forward_sanity(G, D)
    loss_sanity(G, D)
    comparison_table(g_stats["params"], g_stats["macs"])
    benchmark_section(G)

    _print_header("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
