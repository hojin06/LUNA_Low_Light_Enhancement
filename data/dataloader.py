"""LOL DataLoader 팩토리.

학습/평가용 DataLoader 를 일관된 옵션으로 생성. Windows 환경에서 spawn
오버헤드를 고려해 num_workers 는 적당히 낮게(2~4) 권장.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import LOLDataset


def _pin_memory_ok(user_choice: bool) -> bool:
    """CUDA 가 없으면 pin_memory 는 무의미하므로 강제로 끔."""
    return bool(user_choice) and torch.cuda.is_available()


def get_train_loader(
    data_root: str | os.PathLike,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 256,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    seed: Optional[int] = None,
    full_resize: bool = False,
) -> DataLoader:
    """학습용 DataLoader.

    - 데이터셋:   ``our485`` (485 페어)
    - augmentation: 활성 (paired geometric + low-only photometric)
    - full_resize=True 면 random crop 없이 전체 이미지 리사이즈 사용.
    """
    dataset = LOLDataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
        augment=True,
        full_resize=full_resize,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=_pin_memory_ok(pin_memory),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def get_eval_loader(
    data_root: str | os.PathLike,
    batch_size: int = 1,
    num_workers: int = 2,
    image_size: int = 256,
    pin_memory: bool = True,
) -> DataLoader:
    """평가용 DataLoader.

    - 데이터셋:   ``eval15`` (15 페어)
    - augmentation: 비활성 (resize 만, 결정론적)
    """
    dataset = LOLDataset(
        data_root=data_root,
        split="eval",
        image_size=image_size,
        augment=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=_pin_memory_ok(pin_memory),
        persistent_workers=num_workers > 0,
    )
