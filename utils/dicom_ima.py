from typing import Dict, Tuple

import numpy as np


def _import_pydicom():
    try:
        import pydicom  # type: ignore
    except Exception as exc:
        raise ImportError(
            "pydicom is required for IMA/DICOM IO. Install it with: pip install pydicom"
        ) from exc
    return pydicom


def _get_rescale_params(ds) -> Tuple[float, float]:
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope == 0:
        slope = 1.0
    return slope, intercept


def _normalize_range_alias(norm_range: str) -> str:
    key = str(norm_range).strip().lower()
    aliases = {
        "-1_1": "-1_1",
        "neg1_1": "-1_1",
        "minus1_1": "-1_1",
        "neg_one_one": "-1_1",
        "0_1": "0_1",
        "zero_one": "0_1",
        "zero_to_one": "0_1",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported norm_range='{norm_range}'. Use one of: '-1_1', '0_1'."
        )
    return aliases[key]


def hu_to_normalized(
    hu: np.ndarray,
    hu_min: float,
    hu_max: float,
    norm_range: str = "-1_1",
) -> np.ndarray:
    hu = np.clip(hu, hu_min, hu_max)
    x01 = (hu - hu_min) / (hu_max - hu_min)
    key = _normalize_range_alias(norm_range)
    if key == "0_1":
        return x01.astype(np.float32)
    return (x01 * 2.0 - 1.0).astype(np.float32)


def normalized_to_hu(
    x: np.ndarray,
    hu_min: float,
    hu_max: float,
    norm_range: str = "-1_1",
) -> np.ndarray:
    key = _normalize_range_alias(norm_range)
    if key == "0_1":
        x = np.clip(x, 0.0, 1.0)
        return (x * (hu_max - hu_min) + hu_min).astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (((x + 1.0) * 0.5) * (hu_max - hu_min) + hu_min).astype(np.float32)


def load_hu_from_dicom(path: str) -> Tuple[np.ndarray, Dict[str, float]]:
    pydicom = _import_pydicom()
    ds = pydicom.dcmread(path)
    pixel = ds.pixel_array.astype(np.float32)
    slope, intercept = _get_rescale_params(ds)
    hu = pixel * slope + intercept
    meta = {
        "slope": slope,
        "intercept": intercept,
        "rows": int(getattr(ds, "Rows", hu.shape[0])),
        "cols": int(getattr(ds, "Columns", hu.shape[1])),
    }
    return hu.astype(np.float32), meta
