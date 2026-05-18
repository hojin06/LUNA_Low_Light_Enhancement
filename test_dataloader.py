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
    # 4) 페어 동기화 sanity check
    # ------------------------------------------------------------------
    # photometric (gamma/brightness/noise) 은 low 에만 적용되어 fill 영역의
    # 픽셀값을 변화시키므로 sync 측정에서 노이즈가 된다.  따라서 photometric
    # 확률을 0 으로 둔 사본을 만들어 순수 기하학적 동기화만 검증한다.
    print(f"[Paired sync sanity]  (photometric off)")
    geo_only_aug = PairedAugment(
        image_size=256, training=True,
        p_gamma=0.0, p_brightness=0.0, p_noise=0.0,
        # 회전/perspective 가 실제로 발동하여 fill 영역이 생기도록 확률↑
        p_rotate=1.0, p_perspective=1.0, p_flip=1.0,
    )
    geo_ds = LOLDataset(
        data_root=data_root, split="train", image_size=256,
        transform=geo_only_aug,
    )
    random.seed(7)
    torch.manual_seed(7)
    lo, hi = geo_ds[idx]

    eps = 1e-3
    mask_lo = (lo <= -1.0 + eps).all(dim=0)  # 검은 fill 영역 마스크
    mask_hi = (hi <= -1.0 + eps).all(dim=0)
    inter = (mask_lo & mask_hi).sum().item()
    union = (mask_lo | mask_hi).sum().item()
    iou = inter / union if union > 0 else 1.0

    # 추가로 픽셀 단위 색상 차이 — 동일 기하 변환이면 두 이미지의 fill 영역
    # 위치는 정확히 같아야 하므로 두 마스크의 차이가 0 에 가까워야 한다.
    sym_diff = (mask_lo ^ mask_hi).sum().item()
    total_pix = mask_lo.numel()
    print(f"  fill-mask IoU(low, high) : {iou:.4f}  (1.0 = perfect sync)")
    print(f"  fill-mask sym-diff ratio : {sym_diff / total_pix:.6f}  (낮을수록 좋음)")
    sync_ok = iou >= 0.95
    print(f"  geometric sync           : {'OK' if sync_ok else 'WARN'}")
    print(HRULE)

    return 0


def _assert_range(t: torch.Tensor, name: str) -> None:
    lo, hi = t.min().item(), t.max().item()
    ok = (lo >= -1.0 - 1e-4) and (hi <= 1.0 + 1e-4)
    flag = "OK" if ok else "FAIL"
    print(f"  range check [{name}]: [-1, 1]  {flag}")


if __name__ == "__main__":
    sys.exit(main())
