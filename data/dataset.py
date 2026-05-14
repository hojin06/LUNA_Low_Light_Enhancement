"""LOL v1 (Wei et al., BMVC 2018) 페어 데이터셋.

폴더 구조 (data_root = ``.../LOLdataset``)::

    LOLdataset/
    ├── eval15/
    │   ├── high/   (정상 조도 GT, 15 장, .png)
    │   └── low/    (저조도 입력,   15 장, .png)
    └── our485/
        ├── high/   (485 장)
        └── low/    (485 장)

매칭 규칙: ``low/<name>.png`` ↔ ``high/<name>.png`` (동일 파일명).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import PairedAugment


class LOLDataset(Dataset):
    """LOL v1 paired (low, high) dataset.

    Parameters
    ----------
    data_root : str | PathLike
        ``LOLdataset`` 폴더 경로 (``eval15``, ``our485`` 의 상위).
    split : str
        ``"train"`` → ``our485``,  ``"eval"`` → ``eval15``.
    image_size : int
        모델 입력 해상도. 기본 256.
    augment : bool
        True 면 학습용 augmentation 적용. 기본 transform 이 ``PairedAugment`` 일
        때만 의미가 있음 (custom ``transform`` 인자를 주면 무시됨).
    transform : Callable | None
        ``(low_PIL, high_PIL) -> (low_tensor, high_tensor)`` 형식의 페어
        변환 함수. None 이면 ``PairedAugment`` 기본 사용.
    """

    SPLIT_TO_FOLDER = {"train": "our485", "eval": "eval15"}

    def __init__(
        self,
        data_root: str | os.PathLike,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform: Optional[Callable[[Image.Image, Image.Image],
                                     Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        if split not in self.SPLIT_TO_FOLDER:
            raise ValueError(
                f"split must be one of {list(self.SPLIT_TO_FOLDER)}, got '{split}'"
            )

        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size

        folder = self.SPLIT_TO_FOLDER[split]
        self.low_dir = self.data_root / folder / "low"
        self.high_dir = self.data_root / folder / "high"
        if not self.low_dir.is_dir() or not self.high_dir.is_dir():
            raise FileNotFoundError(
                f"LOL dataset not found. Expected:\n"
                f"  {self.low_dir}\n  {self.high_dir}"
            )

        self.pairs: List[Tuple[Path, Path]] = self._build_pairs()
        if not self.pairs:
            raise RuntimeError(f"No matched (low, high) pairs in {self.low_dir}")

        self.transform = transform or PairedAugment(
            image_size=image_size, training=augment, full_resize=full_resize,
        )

    # ------------------------------------------------------------------
    def _build_pairs(self) -> List[Tuple[Path, Path]]:
        """파일명으로 (low, high) 페어를 매칭. high 가 없는 low 는 스킵."""
        low_files = sorted(self.low_dir.glob("*.png"))
        pairs: List[Tuple[Path, Path]] = []
        for low_p in low_files:
            high_p = self.high_dir / low_p.name
            if high_p.exists():
                pairs.append((low_p, high_p))
        return pairs

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_p, high_p = self.pairs[idx]
        low_img = Image.open(low_p).convert("RGB")
        high_img = Image.open(high_p).convert("RGB")
        return self.transform(low_img, high_img)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"LOLDataset(root='{self.data_root}', split='{self.split}', "
            f"size={self.image_size}, n_pairs={len(self.pairs)})"
        )
