import numpy as np


def compute_official_ssim(pred: np.ndarray, ref: np.ndarray, data_range: float) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "scikit-image is required for SSIM computation. Install it with: pip install scikit-image"
        ) from exc

    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(ref, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"SSIM expects matching shapes, got {x.shape} and {y.shape}.")

    multichannel = x.ndim == 3 and x.shape[-1] > 1
    try:
        return float(
            structural_similarity(
                x,
                y,
                data_range=float(data_range),
                channel_axis=-1 if multichannel else None,
            )
        )
    except TypeError:
        return float(
            structural_similarity(
                x,
                y,
                data_range=float(data_range),
                multichannel=multichannel,
            )
        )
