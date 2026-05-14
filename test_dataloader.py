"""LOL 데이터 파이프라인 검증 스크립트.

실행 결과
---------
1. 학습/평가 페어 개수
2. 학습/평가 배치 1 개의 (low, high) shape / dtype / 값 범위
3. ``samples/`` 폴더에 시각화 PNG 저장
   - low_raw.png / high_raw.png : 변환 전 (resize 만)
   - low_aug.png / high_aug.png : augmentation 1 회 적용
   - lol_aug_grid.png            : raw + 변환된 샘플 3 회 모음 그리드
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# Windows 콘솔(cp949)에서도 한글/유니코드 출력이 깨지지 않도록 UTF-8 강제.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image

from data import LOLDataset, PairedAugment, get_eval_loader, get_train_loader


def _tensor_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) ∈ [-1, 1] 텐서 → (H, W, 3) uint8 ndarray."""
    arr = ((t.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return arr.cpu().numpy().transpose(1, 2, 0)


def save_tensor_image(t: torch.Tensor, path: Path) -> None:
    """단일 이미지 텐서를 PNG 로 저장."""
    Image.fromarray(_tensor_to_uint8_hwc(t)).save(path)


def save_grid(tensors: list[torch.Tensor], path: Path,
              ncol: int = 2, pad: int = 4, pad_value: int = 255) -> None:
    """텐서 리스트를 ncol 컬럼 그리드 PNG 로 저장 (torchvision 의존 X)."""
    if not tensors:
        return
    _, H, W = tensors[0].shape
    n = len(tensors)
    rows = (n + ncol - 1) // ncol
    canvas = np.full(
        (rows * H + (rows + 1) * pad, ncol * W + (ncol + 1) * pad, 3),
        pad_value, dtype=np.uint8,
    )
    for k, t in enumerate(tensors):
        r, c = divmod(k, ncol)
        y = pad + r * (H + pad)
        x = pad + c * (W + pad)
        canvas[y:y + H, x:x + W] = _tensor_to_uint8_hwc(t)
    Image.fromarray(canvas).save(path)


HRULE = "=" * 82
SUBRULE = "-" * 82

DATA_ROOT_CANDIDATES = [
    Path("../DataSet/LOLdataset"),
    Path("../../DataSet/LOLdataset"),
    Path(r"c:/대학교/Projects/SmallSizePM_GAN_model/DataSet/LOLdataset"),
]


def find_data_root() -> Path:
    for p in DATA_ROOT_CANDIDATES:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "LOLdataset 폴더를 찾을 수 없습니다. 시도한 경로:\n  "
        + "\n  ".join(str(p) for p in DATA_ROOT_CANDIDATES)
    )


def main() -> int:
    data_root = find_data_root()

    print(HRULE)
    print(" LOL Dataset 파이프라인 검증")
    print(HRULE)
    print(f"  data_root  : {data_root}")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  CUDA       : {'available' if torch.cuda.is_available() else 'unavailable'}")
    print(SUBRULE)

    # ------------------------------------------------------------------
    # 1) Loaders & dataset 크기
    # ------------------------------------------------------------------
    # Windows 에서 worker spawn 비용을 피하려고 num_workers=0 로 테스트.
    train_loader = get_train_loader(
        data_root, batch_size=8, num_workers=0, image_size=256, seed=42,
    )
    eval_loader = get_eval_loader(
        data_root, batch_size=1, num_workers=0, image_size=256,
    )
    print(f"[Dataset 크기]")
    print(f"  train (our485) : {len(train_loader.dataset)} pairs")
    print(f"  eval  (eval15) : {len(eval_loader.dataset)} pairs")
    print(SUBRULE)

    # ------------------------------------------------------------------
    # 2) 배치 한 개씩 점검
    # ------------------------------------------------------------------
    low_b, high_b = next(iter(train_loader))
    print(f"[Train batch]")
    print(f"  low  : shape={tuple(low_b.shape)}  dtype={low_b.dtype}"
          f"  range=[{low_b.min().item():+.4f}, {low_b.max().item():+.4f}]"
          f"  mean={low_b.mean().item():+.4f}")
    print(f"  high : shape={tuple(high_b.shape)}  dtype={high_b.dtype}"
          f"  range=[{high_b.min().item():+.4f}, {high_b.max().item():+.4f}]"
          f"  mean={high_b.mean().item():+.4f}")
    _assert_range(low_b, "train.low")
    _assert_range(high_b, "train.high")
    print(SUBRULE)

    low_e, high_e = next(iter(eval_loader))
    print(f"[Eval batch]")
    print(f"  low  : shape={tuple(low_e.shape)}"
          f"  range=[{low_e.min().item():+.4f}, {low_e.max().item():+.4f}]")
    print(f"  high : shape={tuple(high_e.shape)}"
          f"  range=[{high_e.min().item():+.4f}, {high_e.max().item():+.4f}]")
    _assert_range(low_e, "eval.low")
    _assert_range(high_e, "eval.high")
    print(SUBRULE)

    # ------------------------------------------------------------------
    # 3) 시각화 — augmentation 전/후 비교
    # ------------------------------------------------------------------
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)

    raw_ds = LOLDataset(
        data_root=data_root, split="train", image_size=256,
        transform=PairedAugment(image_size=256, training=False),
    )
    aug_ds = LOLDataset(
        data_root=data_root, split="train", image_size=256,
        transform=PairedAugment(image_size=256, training=True),
    )

    idx = 0
    low_raw, high_raw = raw_ds[idx]

    # 단일 이미지 저장
    save_tensor_image(low_raw,  samples_dir / "low_raw.png")
    save_tensor_image(high_raw, samples_dir / "high_raw.png")

    # augmentation 적용 결과 1 회 저장
    random.seed(42)
    torch.manual_seed(42)
    low_aug1, high_aug1 = aug_ds[idx]
    save_tensor_image(low_aug1,  samples_dir / "low_aug.png")
    save_tensor_image(high_aug1, samples_dir / "high_aug.png")

    # 비교 그리드: row1 = raw(low|high), row2~4 = augmented samples
    rows_list = [low_raw, high_raw, low_aug1, high_aug1]
    for k in range(2):
        low_k, high_k = aug_ds[idx]
        rows_list.extend([low_k, high_k])
    save_grid(rows_list, samples_dir / "lol_aug_grid.png", ncol=2, pad=4)

    print(f"[Visualization]")
    print(f"  saved to: {samples_dir.resolve()}")
    print(f"   - low_raw.png  / high_raw.png   (resize-only)")
    print(f"   - low_aug.png  / high_aug.png   (after augmentation)")
    print(f"   - lol_aug_grid.png              (raw + 3 aug samples grid)")
    print(SUBRULE)

    # ------------------------------------------------------------------
    # 4) 페어 동기화 sanity check (geometric)
    # ------------------------------------------------------------------
    # 실제 LOL 이미지로는 (a) 자연적 어두운 픽셀, (b) photometric augmentation
    # 이 측정에 섞여 들어가 sync 만 깔끔히 분리해 보기 어렵다.
    # → 동일한 합성 패턴을 low / high 양쪽에 넣고 **photometric 을 끈** 페어
    #   변환을 적용한다.  지오메트릭 변환이 동일하다면 두 출력은 픽셀 단위로
    #   완전히 동일해야 한다.
    print(f"[Paired sync sanity]  (synthetic input, photometric off)")
    from PIL import Image as PILImage

    pattern = PILImage.new("RGB", (600, 400))
    px = pattern.load()
    for y in range(400):
        for x in range(600):
            px[x, y] = (x % 256, y % 256, (x + y) % 256)

    geo_only = PairedAugment(
        image_size=256, training=True,
        p_flip=1.0, p_rotate=1.0, p_perspective=1.0,
        p_gamma=0.0, p_brightness=0.0, p_noise=0.0,
    )
    random.seed(7)
    torch.manual_seed(7)
    lo, hi = geo_only(pattern, pattern.copy())

    max_diff = (lo - hi).abs().max().item()
    mean_diff = (lo - hi).abs().mean().item()
    sync_ok = max_diff < 1e-6
    print(f"  max  |low - high|  : {max_diff:.2e}")
    print(f"  mean |low - high|  : {mean_diff:.2e}")
    print(f"  geometric sync     : {'OK' if sync_ok else 'FAIL'}"
          f"  (동일 합성 입력이면 픽셀 단위 일치 기대)")
    print(HRULE)

    return 0


def _assert_range(t: torch.Tensor, name: str) -> None:
    lo, hi = t.min().item(), t.max().item()
    ok = (lo >= -1.0 - 1e-4) and (hi <= 1.0 + 1e-4)
    flag = "OK" if ok else "FAIL"
    print(f"  range check [{name}]: [-1, 1]  {flag}")


if __name__ == "__main__":
    sys.exit(main())
