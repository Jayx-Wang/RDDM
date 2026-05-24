"""
Run RDDM inference on one LDCT slice, optionally with an NDCT reference.

Model predicts residual in normalized space:
  r_hat = tanh(f_theta(eps, x)),
  y_hat = clamp(x + r_hat, -1, 1).
"""

import argparse
import glob
import os
import random
import re
from pathlib import Path

import numpy as np
import torch as th

from utils import logger
from utils.dicom_ima import hu_to_normalized, load_hu_from_dicom, normalized_to_hu
from utils.metric_util import compute_official_ssim
from utils.script_util import MODEL_ARG_HELP, add_dict_to_argparser, args_to_dict, create_model

try:
    from torch.amp import autocast as torch_autocast

    _HAS_TORCH_AMP = True
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast as torch_autocast  # type: ignore

    _HAS_TORCH_AMP = False


def model_defaults():
    return dict(
        image_size=512,
        in_channels=2,
        num_channels=64,
        out_channels=1,
        num_res_blocks=2,
        num_heads=1,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="",
        channel_mult="",
        dropout=0.0,
        dims=2,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    )


INFER_ARG_HELP = {
    "ldct_path": "Path to one LDCT .IMA/.dcm slice used as the denoising input.",
    "ndct_path": "Optional paired NDCT .IMA/.dcm slice for reporting HU-space metrics.",
    "hu_min": "Lower HU bound used to normalize CT values before inference.",
    "hu_max": "Upper HU bound used to normalize CT values before inference.",
    "window_center": "Window center used only for saved TIFF visualization.",
    "window_width": "Window width used only for saved TIFF visualization.",
    "use_fp16": "Enable mixed-precision inference when CUDA is available.",
    "checkpoint": "Path to a specific .pth checkpoint file. Preferred for pretrained weights with custom filenames.",
    "checkpointdir": "Directory containing model_*.pth checkpoints. The latest file is used if --checkpoint is empty.",
    "use_ema": "Load EMA weights from the checkpoint when available.",
    "enable_stats": "If true, run num_test_samples stochastic passes and report metric statistics against NDCT.",
    "num_test_samples": "Number of stochastic noise samples used when enable_stats=true; otherwise one pass is used.",
    "save_images": "Save windowed TIFF images for LDCT, denoised output, NDCT, and absolute difference when available.",
    "out_dir": "Directory where single-slice visualization images and info text are saved.",
    "seed": "Random seed for Gaussian noise generation and reproducibility.",
}


def create_argparser():
    defaults = dict(
        ldct_path="",
        ndct_path="",
        hu_min=-1024.0,
        hu_max=3072.0,
        window_center=40.0,
        window_width=400.0,
        use_fp16=False,
        checkpoint="",
        checkpointdir="",
        use_ema=True,
        enable_stats=False,
        num_test_samples=8,
        save_images=True,
        out_dir="outputs/single",
        seed=0,
    )
    defaults.update(model_defaults())
    help_dict = {}
    help_dict.update(INFER_ARG_HELP)
    help_dict.update(MODEL_ARG_HELP)
    parser = argparse.ArgumentParser(
        description="Run RDDM inference on a single LDCT slice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dict_to_argparser(parser, defaults, help_dict=help_dict)
    return parser


def _find_latest_checkpoint(checkpointdir: str) -> str:
    ckpts = glob.glob(os.path.join(checkpointdir, "model_*.pth"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint found in {checkpointdir}")
    ckpts.sort()
    return ckpts[-1]


def _load_model_state(model, state, use_ema: bool = True):
    if use_ema and "ema" in state:
        for ema_param, model_param in zip(state["ema"], model.parameters()):
            model_param.data.copy_(ema_param.data.to(model_param.device))
        return
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)


def _safe_torch_load(path: str):
    try:
        return th.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return th.load(path, map_location="cpu")


def _hu_to_uint16_tiff(hu: np.ndarray, wc: float, ww: float) -> np.ndarray:
    if ww <= 0:
        raise ValueError("window_width must be > 0.")
    low = wc - ww / 2.0
    high = wc + ww / 2.0
    clipped = np.clip(hu, low, high)
    scaled = (clipped - low) / (high - low)
    return np.clip(scaled * 65535.0, 0, 65535).round().astype(np.uint16)



def _save_image(path: str, arr: np.ndarray):
    try:
        from PIL import Image
    except Exception as exc:
        raise ImportError("Pillow is required to save TIFF. Install: pip install pillow") from exc
    Image.fromarray(arr).save(path)


def _compute_basic_metrics(pred_hu: np.ndarray, ref_hu: np.ndarray):
    diff = pred_hu.astype(np.float32) - ref_hu.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def _full_hu_data_range(hu_min: float, hu_max: float) -> float:
    hu_min = float(hu_min)
    hu_max = float(hu_max)
    if not hu_min < hu_max:
        raise ValueError(f"Invalid HU range: [{hu_min}, {hu_max}]")
    return hu_max - hu_min


def _compute_psnr(pred: np.ndarray, ref: np.ndarray, data_range: float) -> float:
    mse = float(np.mean((pred.astype(np.float64) - ref.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def _compute_ssim(pred: np.ndarray, ref: np.ndarray, data_range: float) -> float:
    return compute_official_ssim(pred, ref, data_range=data_range)


def _summary_stats(values: np.ndarray):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _extract_patient_slice(path: Path):
    name = path.name
    patient_match = re.search(r"(L\d{3})", name, flags=re.IGNORECASE)
    slice_match = re.search(r"\.CT\.\d+\.(\d{4})\.", name, flags=re.IGNORECASE)

    patient_id = patient_match.group(1).upper() if patient_match else "PAT"
    if slice_match:
        slice_id = slice_match.group(1)
    else:
        all_4 = re.findall(r"(\d{4})", name)
        slice_id = all_4[-1] if len(all_4) > 0 else "SLICE"
    return patient_id, slice_id


def main():
    args = create_argparser().parse_args()
    logger.configure()

    if not args.ldct_path:
        raise ValueError("--ldct_path is required.")
    ldct_path = Path(args.ldct_path)
    if not ldct_path.exists():
        raise FileNotFoundError(f"LDCT file not found: {ldct_path}")

    ndct_path = Path(args.ndct_path) if args.ndct_path else None
    if ndct_path is not None and not ndct_path.exists():
        logger.log(f"[WARN] NDCT path does not exist, ignore: {ndct_path}")
        ndct_path = None

    if not args.checkpoint and not args.checkpointdir:
        raise ValueError("Provide either --checkpoint or --checkpointdir.")
    ckpt_path = args.checkpoint if args.checkpoint else _find_latest_checkpoint(args.checkpointdir)

    if int(args.num_test_samples) <= 0:
        raise ValueError("--num_test_samples must be >= 1.")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    th.manual_seed(int(args.seed))
    if th.cuda.is_available():
        th.cuda.manual_seed_all(int(args.seed))

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    logger.log("creating model...")
    model = create_model(
        **args_to_dict(
            args,
            [
                "image_size",
                "in_channels",
                "num_channels",
                "out_channels",
                "num_res_blocks",
                "dims",
                "channel_mult",
                "class_cond",
                "use_checkpoint",
                "attention_resolutions",
                "num_heads",
                "num_head_channels",
                "num_heads_upsample",
                "use_scale_shift_norm",
                "dropout",
                "resblock_updown",
                "use_new_attention_order",
            ],
        )
    )
    model.to(device)
    model.eval()

    logger.log(f"loading checkpoint: {ckpt_path}")
    state = _safe_torch_load(ckpt_path)
    _load_model_state(model, state, use_ema=bool(args.use_ema))

    ldct_hu, _ = load_hu_from_dicom(str(ldct_path))
    ldct_norm = hu_to_normalized(ldct_hu, args.hu_min, args.hu_max)
    ldct_tensor = th.from_numpy(ldct_norm).to(th.float32)[None, None].to(device)

    ndct_hu = None
    if ndct_path is not None:
        ndct_hu, _ = load_hu_from_dicom(str(ndct_path))

    if _HAS_TORCH_AMP:
        amp_ctx = torch_autocast("cuda", enabled=(args.use_fp16 and th.cuda.is_available()))
    elif args.use_fp16:
        amp_ctx = torch_autocast(enabled=True)
    else:
        from contextlib import nullcontext

        amp_ctx = nullcontext()

    num_test_samples = int(args.num_test_samples) if bool(args.enable_stats) else 1

    denoised_samples = []
    mae_list = []
    rmse_list = []
    psnr_list = []
    ssim_list = []
    metric_range = _full_hu_data_range(args.hu_min, args.hu_max)

    for _ in range(num_test_samples):
        noise = th.randn_like(ldct_tensor)
        with amp_ctx:
            pred = model(noise, x_prime=ldct_tensor)
            residual = th.tanh(pred)
            denoised_norm = (ldct_tensor + residual).clamp(-1.0, 1.0)

        denoised_hu_i = normalized_to_hu(
            denoised_norm.detach().to(th.float32).cpu().numpy()[0, 0],
            args.hu_min,
            args.hu_max,
        )
        denoised_samples.append(denoised_hu_i)

        if ndct_hu is not None and ndct_hu.shape == denoised_hu_i.shape:
            mae_i, rmse_i = _compute_basic_metrics(denoised_hu_i, ndct_hu)
            psnr_i = _compute_psnr(denoised_hu_i, ndct_hu, data_range=metric_range)
            ssim_i = _compute_ssim(denoised_hu_i, ndct_hu, data_range=metric_range)
            mae_list.append(mae_i)
            rmse_list.append(rmse_i)
            psnr_list.append(psnr_i)
            ssim_list.append(ssim_i)

    denoised_stack = np.stack(denoised_samples, axis=0).astype(np.float32)
    denoised_hu = denoised_stack.mean(axis=0)

    wc = float(args.window_center)
    ww = float(args.window_width)

    mae_stats = None
    rmse_stats = None
    psnr_stats = None
    ssim_stats = None
    if ndct_hu is not None:
        if ndct_hu.shape == denoised_hu.shape:
            if len(mae_list) > 0:
                mae_stats = _summary_stats(np.asarray(mae_list))
                rmse_stats = _summary_stats(np.asarray(rmse_list))
                psnr_stats = _summary_stats(np.asarray(psnr_list))
                ssim_stats = _summary_stats(np.asarray(ssim_list))
            logger.log(
                f"[METRIC][HU_FULL] samples={num_test_samples} data_range={metric_range:.6f} "
                f"MAE(mean/std/min/max)="
                f"{mae_stats['mean']:.6f}/{mae_stats['std']:.6f}/{mae_stats['min']:.6f}/{mae_stats['max']:.6f}, "
                f"RMSE(mean/std/min/max)="
                f"{rmse_stats['mean']:.6f}/{rmse_stats['std']:.6f}/{rmse_stats['min']:.6f}/{rmse_stats['max']:.6f}, "
                f"PSNR(mean/std/min/max)="
                f"{psnr_stats['mean']:.6f}/{psnr_stats['std']:.6f}/{psnr_stats['min']:.6f}/{psnr_stats['max']:.6f}, "
                f"SSIM(mean/std/min/max)="
                f"{ssim_stats['mean']:.6f}/{ssim_stats['std']:.6f}/{ssim_stats['min']:.6f}/{ssim_stats['max']:.6f}"
            )
        else:
            logger.log(
                f"[WARN] shape mismatch: denoised={denoised_hu.shape}, ndct={ndct_hu.shape}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_id, slice_id = _extract_patient_slice(ldct_path)
    base = f"{patient_id}_{slice_id}"

    if bool(args.save_images):
        _save_image(str(out_dir / f"{base}_ldct.tiff"), _hu_to_uint16_tiff(ldct_hu, wc, ww))
        _save_image(
            str(out_dir / f"{base}_denoised.tiff"),
            _hu_to_uint16_tiff(denoised_hu, wc, ww),
        )
        if ndct_hu is not None:
            _save_image(str(out_dir / f"{base}_ndct.tiff"), _hu_to_uint16_tiff(ndct_hu, wc, ww))
            absdiff = np.abs(denoised_hu - ndct_hu)
            _save_image(
                str(out_dir / f"{base}_absdiff.tiff"),
                np.clip(absdiff, 0, 65535).astype(np.uint16),
            )

    info_path = out_dir / f"{base}_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint={ckpt_path}\n")
        f.write(f"ldct_path={ldct_path}\n")
        f.write(f"ndct_path={ndct_path if ndct_path is not None else 'N/A'}\n")
        f.write(f"window_center={wc}\n")
        f.write(f"window_width={ww}\n")
        f.write(f"enable_stats={bool(args.enable_stats)}\n")
        f.write(f"num_test_samples={num_test_samples}\n")
        f.write(f"save_images={bool(args.save_images)}\n")
        f.write("metric_space=full_hu\n")
        f.write(f"metric_data_range={metric_range:.6f}\n")
        if mae_stats is not None and rmse_stats is not None:
            f.write(f"mae_hu_mean={mae_stats['mean']:.6f}\n")
            f.write(f"mae_hu_std={mae_stats['std']:.6f}\n")
            f.write(f"mae_hu_min={mae_stats['min']:.6f}\n")
            f.write(f"mae_hu_max={mae_stats['max']:.6f}\n")
            f.write(f"rmse_hu_mean={rmse_stats['mean']:.6f}\n")
            f.write(f"rmse_hu_std={rmse_stats['std']:.6f}\n")
            f.write(f"rmse_hu_min={rmse_stats['min']:.6f}\n")
            f.write(f"rmse_hu_max={rmse_stats['max']:.6f}\n")
        if psnr_stats is not None and ssim_stats is not None:
            f.write(f"psnr_hu_mean={psnr_stats['mean']:.6f}\n")
            f.write(f"psnr_hu_std={psnr_stats['std']:.6f}\n")
            f.write(f"psnr_hu_min={psnr_stats['min']:.6f}\n")
            f.write(f"psnr_hu_max={psnr_stats['max']:.6f}\n")
            f.write(f"ssim_hu_mean={ssim_stats['mean']:.6f}\n")
            f.write(f"ssim_hu_std={ssim_stats['std']:.6f}\n")
            f.write(f"ssim_hu_min={ssim_stats['min']:.6f}\n")
            f.write(f"ssim_hu_max={ssim_stats['max']:.6f}\n")

    logger.log(f"saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
