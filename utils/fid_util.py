from pathlib import Path

import numpy as np

from utils.dicom_ima import hu_to_normalized


def save_hu_batch_as_fid_pngs(
    hu_batch,
    out_dir: Path,
    hu_min: float,
    hu_max: float,
    prefix: str,
    start_index: int = 0,
) -> int:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Pillow is required for FID image staging. Install it with: pip install pillow"
        ) from exc

    arr = np.asarray(hu_batch, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"Expected HU batch with shape [B, H, W] or [H, W], got {arr.shape}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    images_01 = hu_to_normalized(arr, float(hu_min), float(hu_max), norm_range="0_1")
    images_u8 = np.clip(images_01 * 255.0, 0, 255).round().astype(np.uint8)

    for idx, image in enumerate(images_u8):
        path = out_dir / f"{prefix}_{start_index + idx:06d}.png"
        Image.fromarray(image, mode="L").save(path)

    return int(images_u8.shape[0])
