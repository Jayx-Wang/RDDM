from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .augment import AugmentPipe
from utils.dicom_ima import hu_to_normalized, load_hu_from_dicom


def augment_defaults():
    return dict(
        xflip=1,
        yflip=1,
        rotate_int=1,
        translate_int=1,
        scale=1,
        rotate_frac=0,
        aniso=0,
        translate_frac=0,
        brightness=0,
        contrast=0,
        lumaflip=0,
        hue=0,
        saturation=0,
    )


def _collect_ima_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".ima", ".dcm"}
        ]
    else:
        files = [
            path
            for path in root.glob("*")
            if path.is_file() and path.suffix.lower() in {".ima", ".dcm"}
        ]
    return sorted(files, key=lambda x: x.as_posix())


def _resolve_split_root(root: Path, split: str) -> Path:
    split = str(split or "").strip()
    candidates = []
    if split:
        split_root = root / split
        candidates.append(split_root)
        if (split_root / "ldct").exists():
            return split_root

    candidates.append(root)
    if (root / "ldct").exists():
        return root

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate an LDCT directory. Checked: "
        f"{checked}. Expected either <root>/ldct or <root>/<split>/ldct."
    )


class MayoIMADataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        use_cond: bool = True,
        recursive: bool = True,
        hu_min: float = -1024.0,
        hu_max: float = 3072.0,
        norm_range: str = "-1_1",
        return_ndct: bool = False,
        **augment_kwargs,
    ):
        self.root = Path(root)
        self.split = split
        self.use_cond = use_cond
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        norm_key = str(norm_range).strip().lower()
        if norm_key in {"0_1", "zero_one", "zero_to_one"}:
            self.norm_range = "0_1"
            self.norm_low = 0.0
            self.norm_high = 1.0
        elif norm_key in {"-1_1", "neg1_1", "minus1_1", "neg_one_one"}:
            self.norm_range = "-1_1"
            self.norm_low = -1.0
            self.norm_high = 1.0
        else:
            raise ValueError(
                f"Unsupported norm_range='{norm_range}'. Use one of: '-1_1', '0_1'."
            )
        self.return_ndct = return_ndct

        self.split_root = _resolve_split_root(self.root, split)
        self.ldct_root = self.split_root / "ldct"
        self.ndct_root = self.split_root / "ndct"

        if not self.ldct_root.exists():
            raise FileNotFoundError(f"LDCT directory not found: {self.ldct_root}")
        if split in {"train", "val"} and not self.ndct_root.exists():
            raise FileNotFoundError(f"NDCT directory not found: {self.ndct_root}")

        self.samples: List[Tuple[str, Optional[str], str]] = []
        ldct_files = _collect_ima_files(self.ldct_root, recursive=recursive)
        if len(ldct_files) == 0:
            raise FileNotFoundError(f"No IMA files found under: {self.ldct_root}")

        for ldct_path in ldct_files:
            rel_path = ldct_path.relative_to(self.ldct_root)
            ndct_path = self.ndct_root / rel_path
            ndct_exists = ndct_path.exists()
            if split in {"train", "val"} and not ndct_exists:
                raise FileNotFoundError(
                    f"Missing NDCT pair for LDCT file: {ldct_path} -> expected {ndct_path}"
                )
            if return_ndct and not ndct_exists:
                raise FileNotFoundError(
                    f"return_ndct=True but NDCT file is missing: {ndct_path}"
                )
            self.samples.append(
                (
                    str(ldct_path),
                    str(ndct_path) if ndct_exists else None,
                    rel_path.as_posix(),
                )
            )

        self.augmenter = AugmentPipe(**augment_kwargs)
        self.enable_augment = split == "train"

    def __len__(self) -> int:
        return len(self.samples)

    def _load_norm_tensor(self, dicom_path: str) -> torch.Tensor:
        hu, _ = load_hu_from_dicom(dicom_path)
        x = hu_to_normalized(hu, self.hu_min, self.hu_max, norm_range=self.norm_range)
        tensor = torch.from_numpy(x).to(torch.float32).unsqueeze(0)
        tensor.clamp_(self.norm_low, self.norm_high)
        return tensor

    def __getitem__(self, index: int):
        ldct_path, ndct_path, rel_path = self.samples[index]
        ldct = self._load_norm_tensor(ldct_path)
        ndct = self._load_norm_tensor(ndct_path) if ndct_path is not None else None

        if self.enable_augment and ndct is not None:
            ndct_b = ndct[None, ...]
            ldct_b = ldct[None, ...]
            ndct_b, ldct_b = self.augmenter(ndct_b, ldct_b)
            ndct = ndct_b.squeeze(0)
            ldct = ldct_b.squeeze(0)
            ndct.clamp_(self.norm_low, self.norm_high)
            ldct.clamp_(self.norm_low, self.norm_high)

        if self.split in {"train", "val"}:
            if ndct is None:
                raise RuntimeError(f"NDCT is required for split='{self.split}'.")
            if self.use_cond:
                return ndct, {"x_prime": ldct}
            return ndct, {}

        out_rel = str(Path(rel_path).with_suffix(".IMA"))
        meta: Dict[str, str] = {"ldct_path": ldct_path, "rel_path": out_rel}
        if self.return_ndct and ndct is not None:
            return ldct, ndct, meta
        return ldct, meta
