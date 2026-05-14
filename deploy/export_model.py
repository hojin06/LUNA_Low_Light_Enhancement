"""hybrid_v1 → ONNX 변환 + 양자화 (FP16 / INT8) 통합 스크립트.

흐름 (Pipeline)
---------------
1. PyTorch 체크포인트 로드 → hybrid_v1 Generator 재구성
2. torch.onnx.export (opset 17, dynamic batch axis) → ``light_enhance_gan_fp32.onnx``
3. 검증: 동일 랜덤 입력에 대해 PyTorch vs ONNX 출력의 max/mean abs diff 계산
4. FP16 변환: ``onnxconverter_common.float16.convert_float_to_float16``
5. INT8 동적 양자화: ``onnxruntime.quantization.quantize_dynamic``
6. ``deploy/benchmark.py`` 자동 호출 → PSNR/SSIM/FPS/Size 비교표 출력

설계 메모
---------
* **dynamic batch**: 학습은 batch=8, 평가는 batch=1, 임베디드 deploy 는 batch=1
  로 다르기 때문에 batch 축을 dynamic 으로 지정.
* **opset 17**: PyTorch 2.5 / ONNX 1.16 / onnxruntime 1.17+ 가 모두 안전하게
  지원하는 최신 안정 opset. ``F.interpolate(scale_factor=2, bilinear)`` 도
  Resize op 로 정상 변환됨.
* **INT8 dynamic quant**: 가중치만 INT8 화 (활성값은 FP32 유지). 캘리브레이션
  데이터 불필요. Conv-heavy 모델에서는 static quant 대비 속도 이득이 작지만
  사이즈 절감 (~ 4×) 효과는 보장.
* **검증 임계값**: max_abs_diff > 1e-3 이면 WARN, > 1e-2 이면 ERROR 로 표시.

사용 예
-------
.. code-block:: bash

    python deploy/export_model.py \\
        --checkpoint checkpoints/hybrid_v1_stage2_best.pth \\
        --data_root "C:\\대학교\\Projects\\SmallSizePM_GAN_model\\DataSet\\LOLdataset"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# project root 를 import path 에 등록
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Windows 콘솔 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch

from deploy.benchmark import load_pytorch_generator, run_benchmark


HRULE = "=" * 92
SUBRULE = "-" * 92


# ===========================================================================
# 0. 라이브러리 가용성 진단 — 친절한 에러 메시지
# ===========================================================================
def _check_libraries() -> bool:
    """필수 ONNX 라이브러리 가용성 검사. 부재하면 안내 후 False."""
    missing = []
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        missing.append("onnxruntime")
    try:
        from onnxconverter_common import float16  # noqa: F401
    except ImportError:
        missing.append("onnxconverter-common")

    if missing:
        print("[error] 다음 라이브러리가 필요합니다:")
        print(f"    pip install {' '.join(missing)}")
        return False
    return True


# ===========================================================================
# 1. PyTorch → ONNX FP32 변환
# ===========================================================================
def export_to_onnx_fp32(
    G: torch.nn.Module,
    out_path: Path,
    image_size: int = 256,
    opset: int = 17,
) -> bool:
    """PyTorch G → ONNX FP32 변환 (dynamic batch).

    Returns
    -------
    bool : 성공 여부
    """
    print(SUBRULE)
    print(" [1/4] PyTorch → ONNX FP32 변환")
    print(SUBRULE)

    G = G.cpu().eval()  # ONNX export 는 CPU 에서 (CUDA 의존성 없는 그래프)
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.perf_counter()
        torch.onnx.export(
            G, dummy, str(out_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes={
                "input":  {0: "batch"},
                "output": {0: "batch"},
            },
            export_params=True,
        )
        dt = time.perf_counter() - t0
        size_mb = out_path.stat().st_size / (1024 ** 2)
        print(f"  ✓ 변환 성공  ({dt:.1f}s)")
        print(f"    file       : {out_path}")
        print(f"    size       : {size_mb:.3f} MB")
        print(f"    opset      : {opset}")
        print(f"    input shape: (batch, 3, {image_size}, {image_size})  "
              f"(batch axis dynamic)")
        return True
    except Exception as e:
        print(f"  ✗ 변환 실패: {e}")
        return False


# ===========================================================================
# 2. ONNX 모델 검증 (PyTorch vs ONNX 출력 비교)
# ===========================================================================
def validate_onnx(
    G: torch.nn.Module,
    onnx_path: Path,
    image_size: int = 256,
    n_samples: int = 3,
) -> bool:
    """PyTorch G 와 ONNX 모델이 동일 입력에 대해 거의 같은 출력을 내는지 검증.

    임계값
    ------
    * max_abs_diff < 1e-3  → ✓ 통과
    * 1e-3 ≤ max < 1e-2    → ⚠️ 경고 (수치 차이는 있으나 deploy 가능)
    * max ≥ 1e-2           → ✗ 실패 (그래프 구조에 문제 가능)
    """
    print(SUBRULE)
    print(" [2/4] ONNX 모델 검증 (PyTorch vs ONNX 출력 비교)")
    print(SUBRULE)

    import onnxruntime as ort

    G = G.cpu().eval()
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    max_diffs, mean_diffs = [], []
    for i in range(n_samples):
        # 시드 고정으로 재현 가능한 입력 생성
        gen = torch.Generator().manual_seed(42 + i)
        x = torch.randn(1, 3, image_size, image_size, generator=gen,
                        dtype=torch.float32)
        with torch.no_grad():
            y_pt = G(x).numpy()
        y_ort = sess.run([out_name], {in_name: x.numpy()})[0]
        diff = np.abs(y_pt - y_ort)
        max_diffs.append(float(diff.max()))
        mean_diffs.append(float(diff.mean()))
        print(f"  sample {i}: max_abs_diff={diff.max():.3e}, "
              f"mean_abs_diff={diff.mean():.3e}")

    overall_max = max(max_diffs)
    overall_mean = float(np.mean(mean_diffs))
    print(f"  overall   : max={overall_max:.3e}, mean={overall_mean:.3e}")

    if overall_max < 1e-3:
        print("  ✓ 검증 통과 — PyTorch 와 ONNX 출력이 일치합니다.")
        return True
    if overall_max < 1e-2:
        print(f"  ⚠️ 경고 — max_diff={overall_max:.3e} 가 다소 크지만 "
              "deploy 가능 수준입니다.")
        return True
    print(f"  ✗ 검증 실패 — max_diff={overall_max:.3e} (≥ 1e-2). "
          "그래프 변환에 문제가 있을 수 있습니다.")
    return False


# ===========================================================================
# 3. ONNX FP32 → FP16 변환
# ===========================================================================
def convert_to_fp16(
    fp32_path: Path, fp16_path: Path,
) -> bool:
    """FP32 ONNX → FP16 ONNX 변환.

    ``onnxconverter_common.float16.convert_float_to_float16`` 사용.
    keep_io_types=False (기본) 로 입출력도 FP16 으로 변환 → 호출자는 FP16
    입력을 공급해야 함.  (benchmark.py 가 ``is_fp16=True`` 로 처리)
    """
    print(SUBRULE)
    print(" [3/4] ONNX FP32 → FP16 변환")
    print(SUBRULE)

    try:
        import onnx
        from onnxconverter_common import float16

        t0 = time.perf_counter()
        model_fp32 = onnx.load(str(fp32_path))
        model_fp16 = float16.convert_float_to_float16(
            model_fp32,
            keep_io_types=False,    # I/O 도 FP16 (deploy/임베디드 활용도↑)
            disable_shape_infer=False,
        )
        fp16_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model_fp16, str(fp16_path))
        dt = time.perf_counter() - t0

        size_mb = fp16_path.stat().st_size / (1024 ** 2)
        ratio = fp32_path.stat().st_size / max(fp16_path.stat().st_size, 1)
        print(f"  ✓ FP16 변환 성공 ({dt:.1f}s)")
        print(f"    file       : {fp16_path}")
        print(f"    size       : {size_mb:.3f} MB  ({ratio:.2f}× 압축)")
        return True
    except Exception as e:
        print(f"  ✗ FP16 변환 실패: {e}")
        return False


# ===========================================================================
# 4. ONNX FP32 → INT8 동적 양자화 (Dynamic Quantization)
# ===========================================================================
def quantize_to_int8(
    G: torch.nn.Module,
    int8_path: Path,
    image_size: int = 256,
    opset: int = 17,
) -> bool:
    """PyTorch G → INT8 ONNX (dynamic quantization).

    onnxruntime.quantization.quantize_dynamic 은 가중치만 INT8 로 변환하고
    활성값은 런타임에 동적으로 양자화한다. 캘리브레이션 데이터셋이 불필요.

    Conv 연산은 dynamic quant 에서 weight-only quant 만 적용되므로 속도 이득
    은 제한적이지만, 모델 크기는 약 4× 줄어든다.  하드 임베디드 타깃에서는
    sufficient.

    구현 메모
    ---------
    동적 batch 축이 있는 FP32 ONNX 는 onnxruntime 1.20+ 의 symbolic shape
    inference 가 완전히 해결하지 못해 ``quantize_dynamic`` 이 실패한다.
    따라서 INT8 입력용으로는 batch=1 로 고정한 별도 static-batch ONNX 를
    임시로 export 한 뒤 quantize 하고, 임시 파일은 정리한다. INT8 모델
    자체는 inference 시 batch=1 로만 사용되는 것이 일반적이라 동적 batch
    축은 손실이 아니다.
    """
    print(SUBRULE)
    print(" [4/4] ONNX FP32 → INT8 동적 양자화 (Dynamic Quantization)")
    print(SUBRULE)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process

        int8_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        # ---- (a) static-batch FP32 ONNX 를 임시 export ----
        # dynamic_axes 를 비워 batch=1 고정 → symbolic shape inference 가
        # 완벽히 동작.
        static_path = int8_path.with_name("_int8_source_static.onnx")
        G = G.cpu().eval()
        dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
        torch.onnx.export(
            G, dummy, str(static_path),
            input_names=["input"], output_names=["output"],
            opset_version=opset, do_constant_folding=True,
            dynamic_axes=None,   # ★ 정적 shape — INT8 quant 만을 위한 임시 파일
            export_params=True,
        )

        # ---- (b) shape inference + 그래프 최적화 ----
        # onnxruntime ≥ 1.20 의 ``quantize_dynamic`` 은 입력 옆에
        # ``<stem>-inferred.onnx`` 가 있다고 가정하고 항상 그 파일을 읽는다
        # (없으면 FileNotFoundError). ``quant_pre_process`` 가 정확히 그
        # 규약대로 inferred 파일을 만들어준다.
        inferred_path = static_path.with_name(static_path.stem + "-inferred.onnx")
        quant_pre_process(
            input_model=str(static_path),
            output_model_path=str(inferred_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
        )

        # ---- (c) 동적 양자화: weight_type=QInt8 (signed, 대칭) ----
        quantize_dynamic(
            model_input=str(static_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )

        # 임시 파일 모두 정리
        for p in (static_path, inferred_path):
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass
        dt = time.perf_counter() - t0

        size_mb = int8_path.stat().st_size / (1024 ** 2)
        print(f"  ✓ INT8 양자화 성공 ({dt:.1f}s)")
        print(f"    file       : {int8_path}")
        print(f"    size       : {size_mb:.3f} MB")
        print("    note       : weight-only dynamic quant (Conv 는 weight only "
              "INT8). PSNR drop 은 benchmark 단계에서 측정.")
        return True
    except Exception as e:
        print(f"  ✗ INT8 양자화 실패: {e}")
        return False


# ===========================================================================
# 5. CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="hybrid_v1 → ONNX FP32/FP16/INT8 변환 + 자동 벤치마크",
    )
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/hybrid_v1_stage2_best.pth")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="deploy/models")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None,
                   help="PyTorch FP32 평가에 사용할 device "
                        "(기본: cuda 사용 가능 시 cuda)")
    p.add_argument("--skip_benchmark", action="store_true",
                   help="변환만 하고 벤치마크 단계 건너뛰기")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print()
    print(HRULE)
    print(" hybrid_v1 ONNX 변환 + 양자화 + 벤치마크")
    print(HRULE)
    print(f"  checkpoint     : {args.checkpoint}")
    print(f"  data_root      : {args.data_root}")
    print(f"  out_dir        : {args.out_dir}")
    print(f"  image_size     : {args.image_size}")
    print(f"  opset          : {args.opset}")

    # ---- 라이브러리 체크 ----
    if not _check_libraries():
        return 1

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        print(f"[error] 체크포인트가 없습니다: {ckpt_path}")
        return 1

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "light_enhance_gan_fp32.onnx"
    fp16_path = out_dir / "light_enhance_gan_fp16.onnx"
    int8_path = out_dir / "light_enhance_gan_int8.onnx"

    # ---- PyTorch 모델 로드 (CPU 로 — ONNX export 용) ----
    print(SUBRULE)
    print(" [0/4] PyTorch 체크포인트 로드")
    print(SUBRULE)
    G = load_pytorch_generator(ckpt_path, device="cpu")
    n_params = sum(p.numel() for p in G.parameters())
    print(f"  params         : {n_params:,}  ({n_params/1e3:.1f} K)")

    # ---- 1. ONNX FP32 변환 ----
    ok_fp32 = export_to_onnx_fp32(G, fp32_path,
                                  image_size=args.image_size,
                                  opset=args.opset)
    if not ok_fp32:
        return 1

    # ---- 2. 검증 ----
    ok_validate = validate_onnx(G, fp32_path, image_size=args.image_size)
    if not ok_validate:
        # 변환 실패 수준이 아니라 정확도 경고 — 계속 진행하되 사용자에게 알림
        print("  [warn] 검증이 실패했지만 양자화는 계속 진행합니다.")

    # ---- 3. FP16 ----
    ok_fp16 = convert_to_fp16(fp32_path, fp16_path)

    # ---- 4. INT8 ----
    ok_int8 = quantize_to_int8(G, int8_path,
                               image_size=args.image_size,
                               opset=args.opset)

    # ---- 변환 결과 요약 ----
    print()
    print(HRULE)
    print(" 변환 결과 요약")
    print(HRULE)
    print(f"  ONNX FP32  : {'✓ 성공' if ok_fp32 else '✗ 실패'}  ({fp32_path})")
    print(f"  검증       : {'✓ 통과' if ok_validate else '⚠️ 경고/실패'}")
    print(f"  ONNX FP16  : {'✓ 성공' if ok_fp16 else '✗ 실패'}  ({fp16_path})")
    print(f"  ONNX INT8  : {'✓ 성공' if ok_int8 else '✗ 실패'}  ({int8_path})")
    print(HRULE)

    if not ok_fp32:
        return 1

    # ---- 5. 벤치마크 자동 실행 ----
    if not args.skip_benchmark:
        run_benchmark(
            ckpt_path=ckpt_path,
            onnx_fp32=fp32_path if ok_fp32 else None,
            onnx_fp16=fp16_path if ok_fp16 else None,
            onnx_int8=int8_path if ok_int8 else None,
            data_root=args.data_root,
            image_size=args.image_size,
            num_workers=args.num_workers,
            device=args.device,
            save_csv=out_dir / "benchmark_comparison.csv",
            quality_drop_threshold_db=1.0,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
