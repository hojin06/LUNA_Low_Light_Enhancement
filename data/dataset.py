"""LOL 페어 데이터셋 (v1 + v2) + LoLI-Street + CombinedDataset.

지원 데이터셋
-------------
* **LOL v1** (Wei et al., BMVC 2018) — ``LOLDataset``
* **LOL-v2 Real** (Yang et al., TIP 2021) — ``LOLv2RealDataset``
* **LOL-v2 Synthetic** (Yang et al., TIP 2021) — ``LOLv2SyntheticDataset``
* **LoLI-Street** (arXiv 2410.09831, 2024) — ``LoLIStreetDataset``
* **Combined** — 여러 데이터셋을 ConcatDataset 으로 합쳐서 학습 (``CombinedDataset``)

공통 인터페이스
---------------
모든 클래스가 ``(low_tensor, high_tensor)`` ∈ ``[-1, 1]`` 페어를 반환하므로
기존 학습 코드를 그대로 사용 가능. ``PairedAugment`` 로 동일한 augmentation 정책.

매칭 규칙
---------
모든 데이터셋이 ``low/<name>.ext`` ↔ ``high/<name>.ext`` 또는
``Low/<name>.ext`` ↔ ``Normal/<name>.ext`` 의 파일명 일치 페어를 가정.
폴더 이름은 대소문자 무관 + 여러 변형(``low``, ``Low``, ``Low_images``) 자동 탐색.

폴더 구조 (LOL v1, data_root = ``.../LOLdataset``)::

    LOLdataset/
    ├── eval15/
    │   ├── high/   (정상 조도 GT, 15 장, .png)
    │   └── low/    (저조도 입력,   15 장, .png)
    └── our485/
        ├── high/   (485 장)
        └── low/    (485 장)

폴더 구조 (LOL-v2, data_root = ``.../LOL-v2``)::

    LOL-v2/
    ├── Real_captured/
    │   ├── Train/{Low,Normal}/*.png    (~689 페어)
    │   └── Test/{Low,Normal}/*.png     (~100 페어)
    └── Synthetic/
        ├── Train/{Low,Normal}/*.png    (~900 페어)
        └── Test/{Low,Normal}/*.png     (~100 페어)

폴더 구조 (LoLI-Street, data_root = ``.../LoLI-Street``)::

    LoLI-Street/
    ├── train/{low,high}/*.png   (대부분)
    └── test/{low,high}/*.png
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from .augmentation import PairedAugment


# ---------------------------------------------------------------------------
# 공통 유틸리티
# ---------------------------------------------------------------------------
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _find_subdir(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    """``root`` 아래에서 후보 중 처음 존재하는 디렉토리 반환 (대소문자 무관).

    LOL-v2 처럼 fork 마다 ``Low`` / ``low`` / ``Low_images`` 등이 섞이는 경우
    한 번에 해결하기 위한 헬퍼.  존재하지 않으면 None.
    """
    if not root.is_dir():
        return None
    # 정확 매칭 우선
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    # 대소문자 무관 + 부분 매칭 fallback (재귀)
    lower_targets = {c.lower() for c in candidates}
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in lower_targets:
            return p
    return None


def _build_pairs(low_dir: Path, high_dir: Path) -> List[Tuple[Path, Path]]:
    """``low_dir`` / ``high_dir`` 에서 파일명 일치 (low, high) 페어 매칭.

    확장자는 자유 (``.png``, ``.jpg`` 등 모두 가능).  high 가 다른 확장자로
    존재해도 동일 stem 이면 매칭.
    """
    if not low_dir.is_dir() or not high_dir.is_dir():
        return []
    high_index = {p.stem: p for p in high_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS}
    pairs: List[Tuple[Path, Path]] = []
    for low_p in sorted(low_dir.iterdir()):
        if not low_p.is_file() or low_p.suffix.lower() not in _IMG_EXTS:
            continue
        high_p = high_index.get(low_p.stem)
        if high_p is not None:
            pairs.append((low_p, high_p))
    return pairs


# ===========================================================================
# 0. 일반 페어 데이터셋 (base class)
# ===========================================================================
class PairedImageDataset(Dataset):
    """``(low_dir, high_dir)`` 직접 지정 방식의 일반 페어 데이터셋.

    LOL v1/v2, LoLI-Street, 사용자 정의 데이터셋 모두 본 클래스를 상속/wrapping.

    Parameters
    ----------
    low_dir, high_dir : Path
        실제 이미지가 있는 디렉토리 경로 (이미 해결된 절대 경로).
    image_size, augment, full_resize, transform :
        ``LOLDataset`` 과 동일한 의미.
    name : str
        디버그 출력용 데이터셋 이름.
    """

    def __init__(
        self,
        low_dir: Path,
        high_dir: Path,
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform: Optional[Callable[[Image.Image, Image.Image],
                                     Tuple[torch.Tensor, torch.Tensor]]] = None,
        name: str = "PairedImageDataset",
    ) -> None:
        super().__init__()
        self.name = name
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.image_size = image_size

        if not self.low_dir.is_dir() or not self.high_dir.is_dir():
            raise FileNotFoundError(
                f"{name}: low/high 디렉토리를 찾을 수 없습니다.\n"
                f"  low  = {self.low_dir}\n  high = {self.high_dir}"
            )

        self.pairs: List[Tuple[Path, Path]] = _build_pairs(
            self.low_dir, self.high_dir,
        )
        if not self.pairs:
            raise RuntimeError(
                f"{name}: (low, high) 페어가 0 개입니다.  파일명이 일치하지 "
                f"않거나 디렉토리가 비어 있습니다.\n  low  = {self.low_dir}\n"
                f"  high = {self.high_dir}"
            )

        self.transform = transform or PairedAugment(
            image_size=image_size, training=augment, full_resize=full_resize,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_p, high_p = self.pairs[idx]
        low_img = Image.open(low_p).convert("RGB")
        high_img = Image.open(high_p).convert("RGB")
        return self.transform(low_img, high_img)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"n_pairs={len(self.pairs)}, image_size={self.image_size})"
        )


# ===========================================================================
# 1. LOL v1
# ===========================================================================
class LOLDataset(PairedImageDataset):
    """LOL v1 paired (low, high) dataset.

    Parameters
    ----------
    data_root : str | PathLike
        ``LOLdataset`` 폴더 경로 (``eval15``, ``our485`` 의 상위).
    split : str
        ``"train"`` → ``our485``,  ``"eval"`` → ``eval15``.
    image_size, augment, full_resize, transform : 기존과 동일.
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
        if split not in self.SPLIT_TO_FOLDER:
            raise ValueError(
                f"split must be one of {list(self.SPLIT_TO_FOLDER)}, got '{split}'"
            )

        self.data_root = Path(data_root)
        self.split = split
        folder = self.SPLIT_TO_FOLDER[split]
        low_dir  = self.data_root / folder / "low"
        high_dir = self.data_root / folder / "high"

        if not low_dir.is_dir() or not high_dir.is_dir():
            raise FileNotFoundError(
                f"LOL v1 dataset not found. Expected:\n"
                f"  {low_dir}\n  {high_dir}"
            )

        super().__init__(
            low_dir=low_dir, high_dir=high_dir,
            image_size=image_size, augment=augment,
            full_resize=full_resize, transform=transform,
            name=f"LOLv1[{split}]",
        )


# ===========================================================================
# 2. LOL-v2 Real
# ===========================================================================
class LOLv2RealDataset(PairedImageDataset):
    """LOL-v2 Real captured paired dataset (Yang et al., TIP 2021).

    Parameters
    ----------
    data_root : str | PathLike
        ``LOL-v2`` 폴더 경로 (``Real_captured``, ``Synthetic`` 의 상위).
    split : str
        ``"train"`` → ``Real_captured/Train``,
        ``"eval"`` / ``"test"`` → ``Real_captured/Test``.

    저조도 / 정상 폴더명은 다음 후보에서 자동 매칭:
    ``Low`` / ``low`` / ``Low_images``,
    ``Normal`` / ``normal`` / ``Normal_images`` / ``high``.
    """

    SPLIT_TO_FOLDER = {"train": "Train", "eval": "Test", "test": "Test"}

    LOW_CANDIDATES  = ("Low", "low", "Low_images", "LOW")
    HIGH_CANDIDATES = ("Normal", "normal", "Normal_images",
                       "high", "High", "GT", "gt")

    def __init__(
        self,
        data_root: str | os.PathLike,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform: Optional[Callable[[Image.Image, Image.Image],
                                     Tuple[torch.Tensor, torch.Tensor]]] = None,
        subset_dir: str = "Real_captured",
    ) -> None:
        if split not in self.SPLIT_TO_FOLDER:
            raise ValueError(
                f"split must be one of {list(self.SPLIT_TO_FOLDER)}, got '{split}'"
            )
        self.data_root = Path(data_root)
        self.split = split

        split_dir = self.data_root / subset_dir / self.SPLIT_TO_FOLDER[split]
        if not split_dir.is_dir():
            # 대소문자 무관 fallback
            split_dir_found = _find_subdir(
                self.data_root / subset_dir,
                [self.SPLIT_TO_FOLDER[split], self.SPLIT_TO_FOLDER[split].lower()],
            )
            if split_dir_found is None:
                raise FileNotFoundError(
                    f"LOL-v2 Real split 디렉토리 없음: {split_dir}"
                )
            split_dir = split_dir_found

        low_dir  = _find_subdir(split_dir, self.LOW_CANDIDATES)
        high_dir = _find_subdir(split_dir, self.HIGH_CANDIDATES)
        if low_dir is None or high_dir is None:
            raise FileNotFoundError(
                f"LOL-v2 Real 의 low/high 폴더를 찾을 수 없습니다.\n"
                f"  탐색 위치 : {split_dir}\n"
                f"  low 후보  : {self.LOW_CANDIDATES}\n"
                f"  high 후보 : {self.HIGH_CANDIDATES}"
            )

        super().__init__(
            low_dir=low_dir, high_dir=high_dir,
            image_size=image_size, augment=augment,
            full_resize=full_resize, transform=transform,
            name=f"LOLv2-Real[{split}]",
        )


# ===========================================================================
# 3. LOL-v2 Synthetic
# ===========================================================================
class LOLv2SyntheticDataset(LOLv2RealDataset):
    """LOL-v2 Synthetic paired dataset.

    Real 과 동일한 폴더 구조 (Synthetic/{Train,Test}/{Low,Normal}) 를 가지므로
    ``subset_dir`` 만 ``"Synthetic"`` 으로 바꾼 ``LOLv2RealDataset`` 의 재사용.
    """

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
        super().__init__(
            data_root=data_root, split=split,
            image_size=image_size, augment=augment,
            full_resize=full_resize, transform=transform,
            subset_dir="Synthetic",
        )
        # 이름만 재지정
        self.name = f"LOLv2-Syn[{split}]"


# ===========================================================================
# 4. LoLI-Street
# ===========================================================================
class LoLIStreetDataset(PairedImageDataset):
    """LoLI-Street paired dataset (arXiv 2410.09831, 2024).

    33,000 페어 거리 장면 저조도/정상 이미지. 공식 폴더 구조가 fork 마다
    다를 수 있어 ``low_subdir``, ``high_subdir`` 직접 지정도 지원.

    예상 구조 (defensive):
    ``LoLI-Street/{train,test}/{low,high}/*.png``
    """

    SPLIT_TO_FOLDER = {"train": "train", "eval": "test", "test": "test"}

    LOW_CANDIDATES  = ("low", "Low", "input", "dark", "lowlight")
    HIGH_CANDIDATES = ("high", "High", "normal", "Normal", "gt", "GT",
                       "target", "reference")

    def __init__(
        self,
        data_root: str | os.PathLike,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        full_resize: bool = False,
        transform: Optional[Callable[[Image.Image, Image.Image],
                                     Tuple[torch.Tensor, torch.Tensor]]] = None,
        low_subdir: Optional[str] = None,
        high_subdir: Optional[str] = None,
    ) -> None:
        if split not in self.SPLIT_TO_FOLDER:
            raise ValueError(
                f"split must be one of {list(self.SPLIT_TO_FOLDER)}, got '{split}'"
            )
        self.data_root = Path(data_root)
        self.split = split

        # 사용자 명시 경로가 있으면 우선
        if low_subdir is not None and high_subdir is not None:
            low_dir = self.data_root / low_subdir
            high_dir = self.data_root / high_subdir
        else:
            split_dir = _find_subdir(
                self.data_root,
                [self.SPLIT_TO_FOLDER[split],
                 self.SPLIT_TO_FOLDER[split].lower()],
            )
            if split_dir is None:
                # 데이터셋이 train/test 분할 없이 평탄하게 있는 경우 대응
                split_dir = self.data_root

            low_dir  = _find_subdir(split_dir, self.LOW_CANDIDATES)
            high_dir = _find_subdir(split_dir, self.HIGH_CANDIDATES)

        if low_dir is None or high_dir is None:
            raise FileNotFoundError(
                f"LoLI-Street 의 low/high 폴더를 찾을 수 없습니다.\n"
                f"  data_root : {self.data_root}\n"
                f"  split     : {split}\n"
                f"  hint      : low_subdir / high_subdir 인자로 직접 지정 가능."
            )

        super().__init__(
            low_dir=low_dir, high_dir=high_dir,
            image_size=image_size, augment=augment,
            full_resize=full_resize, transform=transform,
            name=f"LoLI-Street[{split}]",
        )


# ===========================================================================
# 5. CombinedDataset — 여러 데이터셋을 합쳐서 학습
# ===========================================================================
class CombinedDataset(ConcatDataset):
    """여러 페어 데이터셋을 ``torch.utils.data.ConcatDataset`` 으로 합침.

    각 데이터셋의 길이를 그대로 사용 (별도의 oversampling 없음).
    필요 시 ``weighted=True`` 옵션을 통해 ``WeightedRandomSampler`` 와 함께
    사용하는 패턴을 권장 (본 클래스는 그 hook 만 제공).

    Parameters
    ----------
    datasets : list[Dataset]
        합칠 데이터셋들. 각각이 ``(low, high)`` 페어를 반환해야 함.
    """

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        if not datasets:
            raise ValueError("CombinedDataset 은 빈 리스트를 받을 수 없습니다.")
        super().__init__(list(datasets))

    def per_dataset_lengths(self) -> List[int]:
        """각 sub-dataset 의 길이 리스트."""
        # ConcatDataset.cumulative_sizes 로 역산
        prev = 0
        out: List[int] = []
        for c in self.cumulative_sizes:
            out.append(c - prev)
            prev = c
        return out

    def __repr__(self) -> str:
        names = []
        for d in self.datasets:
            names.append(getattr(d, "name", d.__class__.__name__))
        total = sum(self.per_dataset_lengths())
        return (f"CombinedDataset(total={total}, "
                f"sub={list(zip(names, self.per_dataset_lengths()))})")


# ===========================================================================
# 6. 통합 팩토리 — 문자열 키 → 데이터셋 인스턴스
# ===========================================================================
DATASET_REGISTRY = {
    "lol_v1":       LOLDataset,
    "lol_v2_real":  LOLv2RealDataset,
    "lol_v2_syn":   LOLv2SyntheticDataset,
    "loli_street":  LoLIStreetDataset,
}


def build_dataset_by_name(
    name: str,
    data_root: str | os.PathLike,
    split: str = "train",
    image_size: int = 256,
    augment: bool = True,
    full_resize: bool = False,
    **kwargs,
) -> Dataset:
    """문자열 키 → 데이터셋 인스턴스.

    ``train_extended.py`` 의 ``--dataset`` CLI 인자를 그대로 매핑.
    알 수 없는 키는 ValueError.
    """
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'.  available: {list(DATASET_REGISTRY)}"
        )
    cls = DATASET_REGISTRY[name]
    return cls(
        data_root=data_root,
        split=split,
        image_size=image_size,
        augment=augment,
        full_resize=full_resize,
        **kwargs,
    )


def build_combined_dataset(
    dataset_root: str | os.PathLike,
    split: str = "train",
    image_size: int = 256,
    augment: bool = True,
    full_resize: bool = False,
    include: Sequence[str] = ("lol_v1", "lol_v2_real",
                              "lol_v2_syn", "loli_street"),
    skip_missing: bool = True,
) -> CombinedDataset:
    """``DataSet/`` 부모 폴더 한 개를 받아 가용한 모든 데이터셋을 자동 합침.

    각 데이터셋의 실제 경로는 부모 폴더 + 표준 디렉토리 이름으로 자동 추정.
    누락된 데이터셋은 ``skip_missing=True`` 면 경고만 출력하고 건너뜀.

    Returns
    -------
    CombinedDataset

    Raises
    ------
    RuntimeError : 단 한 개도 로드되지 않은 경우.
    """
    root = Path(dataset_root)
    # name → (root_subdir, kwargs)
    subdir_map = {
        "lol_v1":       "LOLdataset",
        "lol_v2_real":  "LOL-v2",
        "lol_v2_syn":   "LOL-v2",
        "loli_street":  "LoLI-Street",
    }

    datasets: List[Dataset] = []
    for key in include:
        if key not in subdir_map:
            print(f"[CombinedDataset] WARN: 알 수 없는 키 '{key}' — 건너뜀")
            continue
        ds_root = root / subdir_map[key]
        try:
            ds = build_dataset_by_name(
                name=key, data_root=ds_root, split=split,
                image_size=image_size, augment=augment,
                full_resize=full_resize,
            )
        except (FileNotFoundError, RuntimeError) as e:
            if skip_missing:
                print(f"[CombinedDataset] SKIP {key}: {e}")
                continue
            raise
        datasets.append(ds)
        print(f"[CombinedDataset] loaded {key} ({len(ds)} pairs)")

    if not datasets:
        raise RuntimeError(
            f"CombinedDataset: 로드된 데이터셋이 0 개입니다.  "
            f"dataset_root={dataset_root} 확인."
        )
    return CombinedDataset(datasets)
