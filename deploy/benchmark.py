"""hybrid_v1 PyTorch / ONNX (FP32 / FP16 / INT8) 종합 벤치마크.

평가 항목
---------
* PSNR / SSIM   : LOL eval15 (15 페어, 256×256)
* FPS           : onnxruntime CUDA EP (가능 시) + CPU EP, batch=1
* 파일 크기     : .onnx 파일 또는 .pth 파일의 실제 디스크 사이즈
* 품질 손실 경보: INT8 PSNR 이 PyTorch FP32 대비 1.0 dB 이상 떨어지면 ⚠️

스크립트 단독 사용
------------------
.. code-block:: bash

    python deploy/benchmark.py \\
        --checkpoint checkpoints/hybrid_v1_stage2_best.pth \\
        --data_root "C:\\대학교\\Projects\\SmallSizePM_GAN_model\\DataSet\\LOLdataset"

모듈 import (export_model.py 에서 호출)
---------------------------------------
.. code-block:: python

    from deploy.benchmark import run_benchmark
    run_benchmark(ckpt_path, onnx_fp32, onnx_fp16, onnx_int8, data_root)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# project root 를 import path 에 등록 (data / models / utils 사용)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Windows 콘솔 UTF-8 강제 (한글 안전 출력)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch

from data import get_eval_loader
from models import LightEnhanceGenerator
from utils import psnr_metric, ssim_metric
from utils.model_analysis import benchmark_inference


HRULE = "=" * 92
SUBRULE = "-" * 92


# ===========================================================================
# 1. Helpers
# ===========================================================================
def _file_size_mb(path: Path) -> float:
    """파일 사이즈 (MB). 존재하지 않으면 NaN."""
    if not path.is_file():
        return float("nan")
    return path.stat().st_size / (1024 ** 2)


def _fmt(x: float, digits: int = 2, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and (x != x)):  # None or NaN
        return "—"
    return f"{x:.{digits}f}{suffix}"


def load_pytorch_generator(
    ckpt_path: Path, device: str,
) -> LightEnhanceGenerator:
    """체크포인트로부터 hybrid_v1 G 재구성 + weight 로드.

    체크포인트 내부의 ``conv_config`` / ``base_filters`` / ``use_attention``
    을 그대로 사용 → 학습 시 구조와 항상 일치.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    bf = int(state.get("base_filters", 32))
    use_attention = bool(state.get("use_attention", True))
    conv_config = state.get("conv_config")
    if conv_config is None:
        # legacy 8-블록 layout 으로 fallback (실제로는 사용되지 않을 것)
        G = LightEnhanceGenerator(base_filters=bf, use_attention=use_attention)
    else:
        G = LightEnhanceGenerator(
            base_filters=bf,
            use_attention=use_attention,
            conv_config=conv_config,
        )
    sd = state["generator"] if "generator" in state else state
    G.load_state_dict(sd)
    return G.to(device).eval()


# ===========================================================================
# 2. ONNX Runtime helpers (선택적 import — 라이브러리 없으면 graceful skip)
# ===========================================================================
def _try_import_onnxruntime() -> "Optional[object]":
    """onnxruntime 가용성 검사. 없으면 None."""
    try:
        import onnxruntime as ort  # type: ignore
        return ort
    except ImportError:
        return None


def make_ort_session(
    onnx_path: Path, provider: str,
) -> "Optional[object]":
    """``provider`` ('cuda' or 'cpu') 로 onnxruntime InferenceSession 생성.

    Returns
    -------
    session 또는 None
        CUDAExecutionProvider 가 가용하지 않으면 None 을 반환 (CPU EP 만 있는
        onnxruntime 배포 환경 대비).
    """
    ort = _try_import_onnxruntime()
    if ort is None:
        return None

    available = ort.get_available_providers()  # type: ignore[attr-defined]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            return None
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()  # type: ignore[attr-defined]
    # 모든 그래프 최적화 활성화 (default 와 동일, 명시적 표기용)
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore[attr-defined]
    )
    try:
        return ort.InferenceSession(  # type: ignore[attr-defined]
            str(onnx_path), sess_options=sess_options, providers=providers,
        )
    except Exception as e:
        print(f"    [warn] {onnx_path.name} on {provider}: {e}")
        return None


# ===========================================================================
# 3. ONNX Runtime evaluation (PSNR / SSIM on eval15)
# ===========================================================================
@torch.no_grad()
def eval_onnx_quality(
    session, eval_loader, input_name: str, output_name: str,
    is_fp16: bool = False,
) -> Dict[str, float]:
    """ONNX 세션으로 eval15 전체 추론 → PSNR / SSIM 평균.

    Parameters
    ----------
    is_fp16 : bool
        True 면 입력 numpy 배열을 ``float16`` 으로 캐스팅 (FP16 ONNX 모델 호환).
    """
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    for low, high in eval_loader:
        # low / high 는 PyTorch tensor [-1, 1] 범위, shape (B,3,H,W)
        low_np = low.numpy()
        if is_fp16:
            low_np = low_np.astype(np.float16)
        else:
            low_np = low_np.astype(np.float32)
        out_np = session.run([output_name], {input_name: low_np})[0]
        # PSNR/SSIM 함수는 PyTorch tensor 를 받음 → 항상 float32 로 변환
        out_t = torch.from_numpy(out_np.astype(np.float32)).clamp(-1.0, 1.0)
        bs = low.size(0)
        psnr_sum += psnr_metric(out_t, high) * bs
        ssim_sum += ssim_metric(out_t, high) * bs
        n += bs
    return {
        "psnr": psnr_sum / max(n, 1),
        "ssim": ssim_sum / max(n, 1),
        "n": float(n),
    }


# ===========================================================================
# 4. ONNX Runtime FPS benchmark
# ===========================================================================
def bench_onnx_fps(
    session, input_name: str, output_name: str,
    input_shape: Tuple[int, ...] = (1, 3, 256, 256),
    n_warmup: int = 5, n_runs: int = 30, is_fp16: bool = False,
) -> Dict[str, float]:
    """ONNX Runtime InferenceSession 의 latency / FPS 측정.

    PyTorch 의 ``utils.model_analysis.benchmark_inference`` 와 동일한 프로토콜
    (warmup → 동기화 → N 회 측정 평균).
    """
    dtype = np.float16 if is_fp16 else np.float32
    x = np.random.randn(*input_shape).astype(dtype)

    # warmup
    for _ in range(n_warmup):
        session.run([output_name], {input_name: x})

    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run([output_name], {input_name: x})
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / n_runs
    fps = (1000.0 / avg_ms) * input_shape[0]
    return {"avg_ms": avg_ms, "fps": fps, "runs": float(n_runs)}


# ===========================================================================
# 5. 결과 행 데이터클래스 + 표 출력
# ===========================================================================
@dataclass
class ModelRow:
    name: str
    size_mb: float
    psnr: float
    ssim: float
    fps_gpu: float
    fps_cpu: float
    note: str = ""


def print_comparison_table(rows: List[ModelRow]) -> None:
    print(HRULE)
    print(" Deployment Benchmark — LOL eval15 (256×256, batch=1)")
    print(HRULE)
    hdr = (f"  {'Model':<20} {'Size':>8} {'PSNR↑':>7} {'SSIM↑':>7} "
           f"{'FPS_GPU':>9} {'FPS_CPU':>9}  Notes")
    print(hdr)
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*9}  {'-'*30}")
    for r in rows:
        size_s = _fmt(r.size_mb, 2, "MB")
        print(
            f"  {r.name:<20} {size_s:>8} "
            f"{_fmt(r.psnr, 2):>7} {_fmt(r.ssim, 3):>7} "
            f"{_fmt(r.fps_gpu, 0):>9} {_fmt(r.fps_cpu, 1):>9}  "
            f"{r.note}"
        )
    print(SUBRULE)


# ===========================================================================
# 6. Main benchmark — 4개 모델 측정 + 비교표
# ===========================================================================
def run_benchmark(
    ckpt_path: Path,
    onnx_fp32: Optional[Path],
    onnx_fp16: Optional[Path],
    onnx_int8: Optional[Path],
    data_root: str,
    *,
    image_size: int = 256,
    num_workers: int = 2,
    device: Optional[str] = None,
    n_warmup: int = 5,
    n_runs_gpu: int = 30,
    n_runs_cpu: int = 10,
    save_csv: Optional[Path] = None,
    quality_drop_threshold_db: float = 1.0,
) -> List[ModelRow]:
    """4개 모델 모두 평가 → ``ModelRow`` 리스트 반환.

    None 으로 전달된 ONNX 경로는 건너뜀 (해당 행을 '— not built —' 로 표시).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 데이터 ----
    eval_loader = get_eval_loader(
        data_root, batch_size=1, num_workers=num_workers,
        image_size=image_size,
    )
    n_pairs = len(eval_loader.dataset)  # type: ignore[arg-type]

    print()
    print(HRULE)
    print(" Benchmark 시작")
    print(HRULE)
    print(f"  checkpoint    : {ckpt_path}")
    print(f"  ONNX fp32     : {onnx_fp32}")
    print(f"  ONNX fp16     : {onnx_fp16}")
    print(f"  ONNX int8     : {onnx_int8}")
    print(f"  data_root     : {data_root}")
    print(f"  device(torch) : {device}")
    print(f"  eval15 pairs  : {n_pairs}")
    ort = _try_import_onnxruntime()
    if ort is None:
        print("  [WARN] onnxruntime 미설치 — ONNX 행은 모두 skip.")
    else:
        providers = ort.get_available_providers()  # type: ignore[attr-defined]
        print(f"  ORT providers : {providers}")
    print(SUBRULE)

    rows: List[ModelRow] = []

    # =====================================================================
    # (a) PyTorch FP32
    # =====================================================================
    print(" [1/4] PyTorch FP32 측정 중…")
    G = load_pytorch_generator(ckpt_path, device=device)

    # PSNR/SSIM
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for low, high in eval_loader:
            low_d  = low.to(device, non_blocking=True)
            high_d = high.to(device, non_blocking=True)
            fake = G(low_d).clamp(-1.0, 1.0)
            bs = low.size(0)
            psnr_sum += psnr_metric(fake, high_d) * bs
            ssim_sum += ssim_metric(fake, high_d) * bs
            n += bs
    pt_psnr = psnr_sum / max(n, 1)
    pt_ssim = ssim_sum / max(n, 1)

    # FPS (GPU + CPU)
    fps_gpu = float("nan")
    if device.startswith("cuda"):
        b = benchmark_inference(G, (3, image_size, image_size),
                                device="cuda", n_warmup=n_warmup,
                                n_runs=n_runs_gpu)
        fps_gpu = float(b["fps"])
    G_cpu = G.cpu()
    b_cpu = benchmark_inference(G_cpu, (3, image_size, image_size),
                                device="cpu",
                                n_warmup=max(n_warmup // 2, 1),
                                n_runs=n_runs_cpu)
    fps_cpu = float(b_cpu["fps"])
    # 다시 GPU 로 복원하지 않음 — 이미 측정 끝났고 메모리 절약

    rows.append(ModelRow(
        name="PyTorch FP32",
        size_mb=_file_size_mb(ckpt_path),
        psnr=pt_psnr, ssim=pt_ssim,
        fps_gpu=fps_gpu, fps_cpu=fps_cpu,
        note="reference",
    ))
    print(f"        PSNR={pt_psnr:.2f}  SSIM={pt_ssim:.4f}  "
          f"FPS_GPU={fps_gpu:.0f}  FPS_CPU={fps_cpu:.1f}")

    # =====================================================================
    # (b/c/d) ONNX FP32 / FP16 / INT8
    # =====================================================================
    onnx_jobs = [
        ("ONNX FP32", onnx_fp32, False),
        ("ONNX FP16", onnx_fp16, True),
        ("ONNX INT8", onnx_int8, False),
    ]
    for idx, (name, onnx_path, is_fp16) in enumerate(onnx_jobs, start=2):
        print(f" [{idx}/4] {name} 측정 중…")

        if onnx_path is None or not onnx_path.is_file() or ort is None:
            rows.append(ModelRow(
                name=name,
                size_mb=_file_size_mb(onnx_path) if onnx_path else float("nan"),
                psnr=float("nan"), ssim=float("nan"),
                fps_gpu=float("nan"), fps_cpu=float("nan"),
                note="not built" if onnx_path is None or not (
                    onnx_path and onnx_path.is_file()
                ) else "no onnxruntime",
            ))
            print(f"        skip — 파일/라이브러리 없음")
            continue

        # ---- 품질: CPU EP 로 측정 (CUDA EP 에서 FP16 정확도 동일하므로 CPU 일관) ----
        cpu_sess = make_ort_session(onnx_path, "cpu")
        if cpu_sess is None:
            rows.append(ModelRow(
                name=name, size_mb=_file_size_mb(onnx_path),
                psnr=float("nan"), ssim=float("nan"),
                fps_gpu=float("nan"), fps_cpu=float("nan"),
                note="session failed",
            ))
            continue

        in_name = cpu_sess.get_inputs()[0].name  # type: ignore[attr-defined]
        out_name = cpu_sess.get_outputs()[0].name  # type: ignore[attr-defined]
        q = eval_onnx_quality(cpu_sess, eval_loader,
                              in_name, out_name, is_fp16=is_fp16)

        # ---- FPS_CPU ----
        cpu_fps = bench_onnx_fps(
            cpu_sess, in_name, out_name,
            input_shape=(1, 3, image_size, image_size),
            n_warmup=max(n_warmup // 2, 1), n_runs=n_runs_cpu,
            is_fp16=is_fp16,
        )["fps"]

        # ---- FPS_GPU (CUDA EP 가용 시) ----
        gpu_fps = float("nan")
        gpu_sess = make_ort_session(onnx_path, "cuda")
        if gpu_sess is not None:
            in_name_g  = gpu_sess.get_inputs()[0].name  # type: ignore[attr-defined]
            out_name_g = gpu_sess.get_outputs()[0].name  # type: ignore[attr-defined]
            gpu_fps = bench_onnx_fps(
                gpu_sess, in_name_g, out_name_g,
                input_shape=(1, 3, image_size, image_size),
                n_warmup=n_warmup, n_runs=n_runs_gpu, is_fp16=is_fp16,
            )["fps"]
            # GPU 세션 즉시 해제 (메모리)
            del gpu_sess

        # INT8 dynamic quant 은 Conv 에는 weight-only quant 만 적용되어 CPU 만 의미가 있음
        note = ""
        if name == "ONNX INT8":
            psnr_drop = pt_psnr - q["psnr"]
            if psnr_drop > quality_drop_threshold_db:
                note = (
                    f"⚠️ PSNR drop {psnr_drop:.2f} dB "
                    f"(> {quality_drop_threshold_db:.1f} 임계값)"
                )

        rows.append(ModelRow(
            name=name, size_mb=_file_size_mb(onnx_path),
            psnr=q["psnr"], ssim=q["ssim"],
            fps_gpu=gpu_fps, fps_cpu=cpu_fps, note=note,
        ))
        print(f"        PSNR={q['psnr']:.2f}  SSIM={q['ssim']:.4f}  "
              f"FPS_GPU={gpu_fps:.0f}  FPS_CPU={cpu_fps:.1f}")

    # ---- 표 출력 ----
    print()
    print_comparison_table(rows)

    # ---- 품질 손실 경고 (요약) ----
    print()
    print(" 품질 손실 점검 (PyTorch FP32 기준)")
    print(SUBRULE)
    for r in rows[1:]:
        if r.psnr != r.psnr:  # NaN
            continue
        drop = pt_psnr - r.psnr
        flag = (
            "⚠️ WARN" if (r.name == "ONNX INT8" and drop > quality_drop_threshold_db)
            else "  OK   "
        )
        print(f"  {flag}  {r.name:<14} ΔPSNR = {drop:+.3f} dB")
    print(SUBRULE)

    # ---- CSV 저장 ----
    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with save_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "size_mb", "psnr", "ssim",
                        "fps_gpu", "fps_cpu", "note"])
            for r in rows:
                w.writerow([r.name, f"{r.size_mb:.4f}",
                            f"{r.psnr:.4f}", f"{r.ssim:.4f}",
                            f"{r.fps_gpu:.2f}", f"{r.fps_cpu:.2f}",
                            r.note])
        print(f"  CSV saved → {save_csv}")

    return rows


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="hybrid_v1 PyTorch + ONNX 벤치마크",
    )
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/hybrid_v1_stage2_best.pth")
    p.add_argument("--onnx_dir", type=str, default="deploy/models")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--csv_out", type=str,
                   default="deploy/models/benchmark_comparison.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir).resolve()
    rows = run_benchmark(
        ckpt_path=Path(args.checkpoint).resolve(),
        onnx_fp32=onnx_dir / "light_enhance_gan_fp32.onnx",
        onnx_fp16=onnx_dir / "light_enhance_gan_fp16.onnx",
        onnx_int8=onnx_dir / "light_enhance_gan_int8.onnx",
        data_root=args.data_root,
        image_size=args.image_size,
        num_workers=args.num_workers,
        device=args.device,
        save_csv=Path(args.csv_out).resolve(),
    )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
