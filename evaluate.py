"""
Evaluate a trained RDDM checkpoint on a paired LDCT/NDCT dataset split.

Features:
- noise-conditioned inference
- strict residual reconstruction: y_hat = clamp(x + tanh(f_theta(eps, x)), -1, 1)
- full-HU PSNR/SSIM/MAE/RMSE evaluation against paired NDCT
- per-sample denoised HU export as .npy
- per-sample inference timing and aggregate timing summary
- tqdm progress bar
- saves per-sample CSV and summary JSON/TXT
"""

import argparse
import csv
import glob
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch as th
from tqdm.auto import tqdm

from utils import logger
from utils.dicom_ima import hu_to_normalized, load_hu_from_dicom, normalized_to_hu
from utils.fid_util import save_hu_batch_as_fid_pngs
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


EVAL_ARG_HELP = {
    "data_dir": "Root dataset directory containing <split>/ldct and <split>/ndct subfolders.",
    "split": "Dataset split to evaluate, usually 'test'.",
    "hu_min": "Lower HU bound used to normalize CT values before inference and define metric range.",
    "hu_max": "Upper HU bound used to normalize CT values before inference and define metric range.",
    "use_fp16": "Enable mixed-precision inference when CUDA is available.",
    "checkpoint": "Path to a specific .pth checkpoint file. Preferred for pretrained weights with custom filenames.",
    "checkpointdir": "Directory containing model_*.pth checkpoints. The latest file is used if --checkpoint is empty.",
    "use_ema": "Load EMA weights from the checkpoint when available.",
    "seed": "Random seed for Gaussian noise generation and reproducibility.",
    "num_test_samples": "Number of stochastic denoising passes per slice. Outputs are averaged before metrics/FID.",
    "max_cases": "Maximum number of paired slices to evaluate. 0 evaluates the full split.",
    "compute_fid": "Compute FID between NDCT and denoised outputs using pytorch-fid.",
    "fid_device": "Device used for FID computation. Empty string selects CUDA when available, otherwise CPU.",
    "fid_batch_size": "Batch size used by pytorch-fid.",
    "fid_dims": "Inception feature dimensionality used by pytorch-fid, commonly 2048.",
    "fid_num_workers": "Number of worker processes used by pytorch-fid image loading.",
    "save_npy": "Save each denoised HU slice as a .npy file under out_dir/npy.",
    "out_dir": "Directory where per-sample CSV, summary files, and optional .npy outputs are saved.",
}


def create_argparser():
    defaults = dict(
        data_dir="",
        split="test",
        hu_min=-1024.0,
        hu_max=3072.0,
        use_fp16=False,
        checkpoint="",
        checkpointdir="",
        use_ema=True,
        seed=0,
        num_test_samples=1,
        max_cases=0,
        compute_fid=True,
        fid_device="",
        fid_batch_size=50,
        fid_dims=2048,
        fid_num_workers=1,
        save_npy=True,
        out_dir="outputs/evaluate",
    )
    defaults.update(model_defaults())
    help_dict = {}
    help_dict.update(EVAL_ARG_HELP)
    help_dict.update(MODEL_ARG_HELP)
    parser = argparse.ArgumentParser(
        description="Evaluate an RDDM checkpoint on a paired LDCT/NDCT dataset split.",
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


def _safe_torch_load(path: str):
    try:
        return th.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return th.load(path, map_location="cpu")


def _load_model_state(model, state, use_ema: bool = True):
    if use_ema and "ema" in state:
        for ema_param, model_param in zip(state["ema"], model.parameters()):
            model_param.data.copy_(ema_param.data.to(model_param.device))
        return
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)


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


def _compute_basic_metrics(pred_hu: np.ndarray, ref_hu: np.ndarray):
    diff = pred_hu.astype(np.float32) - ref_hu.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def _summary_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _timed_denoise_step(model, noise, ldct_tensor, amp_context_factory, device: th.device):
    if device.type == "cuda":
        start_event = th.cuda.Event(enable_timing=True)
        end_event = th.cuda.Event(enable_timing=True)
        with amp_context_factory():
            start_event.record()
            pred = model(noise, x_prime=ldct_tensor)
            residual = th.tanh(pred)
            denoised_norm = (ldct_tensor + residual).clamp(-1.0, 1.0)
            end_event.record()
        end_event.synchronize()
        elapsed_sec = float(start_event.elapsed_time(end_event) / 1000.0)
    else:
        start_time = perf_counter()
        with amp_context_factory():
            pred = model(noise, x_prime=ldct_tensor)
            residual = th.tanh(pred)
            denoised_norm = (ldct_tensor + residual).clamp(-1.0, 1.0)
        elapsed_sec = float(perf_counter() - start_time)
    return denoised_norm, elapsed_sec


def _collect_pairs(data_dir: str, split: str):
    ldct_dir = Path(data_dir) / split / "ldct"
    ndct_dir = Path(data_dir) / split / "ndct"
    if not ldct_dir.exists():
        raise FileNotFoundError(f"LDCT directory not found: {ldct_dir}")
    if not ndct_dir.exists():
        raise FileNotFoundError(f"NDCT directory not found: {ndct_dir}")

    ldct_paths = {p.stem: p for p in sorted(ldct_dir.glob("*.IMA"))}
    ndct_paths = {p.stem: p for p in sorted(ndct_dir.glob("*.IMA"))}
    common = sorted(set(ldct_paths.keys()) & set(ndct_paths.keys()))
    if not common:
        raise RuntimeError(f"No paired LDCT/NDCT files found under {ldct_dir} and {ndct_dir}")
    return [(stem, ldct_paths[stem], ndct_paths[stem]) for stem in common]


def main():
    args = create_argparser().parse_args()
    logger.configure()

    if not args.data_dir:
        raise ValueError("--data_dir is required.")
    if not args.checkpoint and not args.checkpointdir:
        raise ValueError("Provide either --checkpoint or --checkpointdir.")
    if int(args.num_test_samples) <= 0:
        raise ValueError("--num_test_samples must be >= 1.")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    th.manual_seed(int(args.seed))
    if th.cuda.is_available():
        th.cuda.manual_seed_all(int(args.seed))

    calculate_fid_given_paths = None
    fid_tmpdir = None
    fid_real_dir = None
    fid_fake_dir = None
    fid_real_count = 0
    fid_fake_count = 0
    if bool(args.compute_fid):
        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths as _calculate_fid_given_paths
        except Exception as exc:
            raise ImportError(
                "pytorch-fid is required for FID computation. Install it with: pip install pytorch-fid"
            ) from exc
        calculate_fid_given_paths = _calculate_fid_given_paths
        fid_tmpdir = TemporaryDirectory(prefix="pytorch_fid_eval_")
        fid_real_dir = Path(fid_tmpdir.name) / "real"
        fid_fake_dir = Path(fid_tmpdir.name) / "fake"
        fid_real_dir.mkdir(parents=True, exist_ok=True)
        fid_fake_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(args.data_dir, args.split)
    if int(args.max_cases) > 0:
        pairs = pairs[: int(args.max_cases)]

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint if args.checkpoint else _find_latest_checkpoint(args.checkpointdir)
    metric_range = _full_hu_data_range(args.hu_min, args.hu_max)

    out_dir = Path(args.out_dir)
    npy_dir = out_dir / "npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.save_npy):
        npy_dir.mkdir(parents=True, exist_ok=True)

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

    if _HAS_TORCH_AMP:
        def amp_context():
            return torch_autocast("cuda", enabled=(args.use_fp16 and th.cuda.is_available()))
    elif args.use_fp16:
        def amp_context():
            return torch_autocast(enabled=True)
    else:
        def amp_context():
            return nullcontext()

    sample_rows = []
    mae_list = []
    rmse_list = []
    psnr_list = []
    ssim_list = []
    infer_times = []

    logger.log(f"running dataset evaluation on {len(pairs)} paired slice(s)...")
    pbar = tqdm(pairs, desc="rddm-evaluate", unit="slice")
    with th.inference_mode():
        for stem, ldct_path, ndct_path in pbar:
            ldct_hu, _ = load_hu_from_dicom(str(ldct_path))
            ndct_hu, _ = load_hu_from_dicom(str(ndct_path))

            ldct_norm = hu_to_normalized(ldct_hu, args.hu_min, args.hu_max)
            ldct_tensor = th.from_numpy(ldct_norm).to(th.float32)[None, None].to(device)

            denoised_samples = []
            sample_infer_times = []
            for _ in range(int(args.num_test_samples)):
                noise = th.randn_like(ldct_tensor)
                denoised_norm, step_infer_time_sec = _timed_denoise_step(
                    model,
                    noise,
                    ldct_tensor,
                    amp_context,
                    device,
                )
                sample_infer_times.append(step_infer_time_sec)
                denoised_hu = normalized_to_hu(
                    denoised_norm.detach().to(th.float32).cpu().numpy()[0, 0],
                    args.hu_min,
                    args.hu_max,
                )
                denoised_samples.append(denoised_hu.astype(np.float32))
            infer_time_sec = float(np.sum(sample_infer_times))
            denoised_hu = np.mean(np.stack(denoised_samples, axis=0), axis=0).astype(np.float32)

            if bool(args.save_npy):
                np.save(str(npy_dir / f"{stem}.npy"), denoised_hu)

            mae, rmse = _compute_basic_metrics(denoised_hu, ndct_hu)
            psnr = _compute_psnr(denoised_hu, ndct_hu, data_range=metric_range)
            ssim = _compute_ssim(denoised_hu, ndct_hu, data_range=metric_range)

            mae_list.append(mae)
            rmse_list.append(rmse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            infer_times.append(infer_time_sec)
            if calculate_fid_given_paths is not None:
                fid_real_count += save_hu_batch_as_fid_pngs(
                    ndct_hu,
                    fid_real_dir,
                    args.hu_min,
                    args.hu_max,
                    prefix="real",
                    start_index=fid_real_count,
                )
                fid_fake_count += save_hu_batch_as_fid_pngs(
                    denoised_hu,
                    fid_fake_dir,
                    args.hu_min,
                    args.hu_max,
                    prefix="fake",
                    start_index=fid_fake_count,
                )

            sample_rows.append(
                {
                    "stem": stem,
                    "ldct_path": str(ldct_path),
                    "ndct_path": str(ndct_path),
                    "npy_path": str(npy_dir / f"{stem}.npy") if bool(args.save_npy) else "",
                    "mae": mae,
                    "rmse": rmse,
                    "psnr": psnr,
                    "ssim": ssim,
                    "infer_time_sec": infer_time_sec,
                }
            )
            pbar.set_postfix(
                psnr=f"{np.mean(psnr_list):.4f}",
                ssim=f"{np.mean(ssim_list):.4f}",
                t=f"{np.mean(infer_times):.4f}s",
            )

    processed = len(sample_rows)
    mae_stats = _summary_stats(mae_list)
    rmse_stats = _summary_stats(rmse_list)
    psnr_stats = _summary_stats(psnr_list)
    ssim_stats = _summary_stats(ssim_list)
    time_stats = _summary_stats(infer_times)
    total_infer_time = float(np.sum(infer_times))
    avg_infer_time = float(np.mean(infer_times))
    try:
        if calculate_fid_given_paths is not None:
            if fid_real_count == 0 or fid_fake_count == 0:
                raise ValueError("FID requires at least one real image and one generated image.")
            fid_device = th.device(
                str(args.fid_device) if args.fid_device else ("cuda" if th.cuda.is_available() else "cpu")
            )
            fid_value = float(
                calculate_fid_given_paths(
                    [str(fid_real_dir), str(fid_fake_dir)],
                    batch_size=int(args.fid_batch_size),
                    device=fid_device,
                    dims=int(args.fid_dims),
                    num_workers=int(args.fid_num_workers),
                )
            )
        else:
            fid_value = None
    finally:
        if fid_tmpdir is not None:
            fid_tmpdir.cleanup()

    logger.log(
        f"[FULLSET][HU_FULL] samples={processed} data_range={metric_range:.6f} "
        f"MAE(mean/std/min/max)="
        f"{mae_stats['mean']:.6f}/{mae_stats['std']:.6f}/{mae_stats['min']:.6f}/{mae_stats['max']:.6f}, "
        f"RMSE(mean/std/min/max)="
        f"{rmse_stats['mean']:.6f}/{rmse_stats['std']:.6f}/{rmse_stats['min']:.6f}/{rmse_stats['max']:.6f}, "
        f"PSNR(mean/std/min/max)="
        f"{psnr_stats['mean']:.6f}/{psnr_stats['std']:.6f}/{psnr_stats['min']:.6f}/{psnr_stats['max']:.6f}, "
        f"SSIM(mean/std/min/max)="
        f"{ssim_stats['mean']:.6f}/{ssim_stats['std']:.6f}/{ssim_stats['min']:.6f}/{ssim_stats['max']:.6f}"
    )
    logger.log(
        f"[FULLSET][TIMING] total_infer_time_sec={total_infer_time:.6f} "
        f"avg_infer_time_per_sample_sec={avg_infer_time:.6f} "
        f"std_infer_time_sec={time_stats['std']:.6f}"
    )
    if fid_value is not None:
        logger.log(f"[FULLSET][FID] value={fid_value:.6f}")

    per_sample_csv = out_dir / "per_sample_metrics.csv"
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "ldct_path",
                "ndct_path",
                "npy_path",
                "mae",
                "rmse",
                "psnr",
                "ssim",
                "infer_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    summary = {
        "checkpoint": ckpt_path,
        "data_dir": args.data_dir,
        "split": args.split,
        "samples": processed,
        "metric_space": "full_hu",
        "metric_data_range": metric_range,
        "mae": mae_stats,
        "rmse": rmse_stats,
        "psnr": psnr_stats,
        "ssim": ssim_stats,
        "timing": {
            "scope": "model_forward_plus_tanh_plus_clamp_only",
            "excludes": [
                "noise_sampling",
                "cpu_transfer",
                "numpy_conversion",
                "hu_denormalization",
                "metric_computation",
                "fid_staging",
                "npy_save",
                "dicom_io",
            ],
            "total_infer_time_sec": total_infer_time,
            "avg_infer_time_per_sample_sec": avg_infer_time,
            "std_infer_time_sec": time_stats["std"],
            "min_infer_time_sec": time_stats["min"],
            "max_infer_time_sec": time_stats["max"],
        },
    }
    if fid_value is not None:
        summary["fid"] = {
            "value": fid_value,
            "backend": "pytorch_fid",
            "space": "full_hu_normalized_0_1_png",
            "dims": int(args.fid_dims),
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"checkpoint={ckpt_path}\n")
        f.write(f"data_dir={args.data_dir}\n")
        f.write(f"split={args.split}\n")
        f.write(f"samples={processed}\n")
        f.write("metric_space=full_hu\n")
        f.write(f"metric_data_range={metric_range:.6f}\n")
        f.write("timing_scope=model_forward_plus_tanh_plus_clamp_only\n")
        f.write(f"mae_mean={mae_stats['mean']:.6f}\n")
        f.write(f"rmse_mean={rmse_stats['mean']:.6f}\n")
        f.write(f"psnr_mean={psnr_stats['mean']:.6f}\n")
        f.write(f"ssim_mean={ssim_stats['mean']:.6f}\n")
        f.write(f"total_infer_time_sec={total_infer_time:.6f}\n")
        f.write(f"avg_infer_time_per_sample_sec={avg_infer_time:.6f}\n")
        f.write(f"std_infer_time_sec={time_stats['std']:.6f}\n")
        if fid_value is not None:
            f.write(f"fid_inception={fid_value:.6f}\n")
            f.write("fid_backend=pytorch_fid\n")
            f.write("fid_space=full_hu_normalized_0_1_png\n")
            f.write(f"fid_dims={int(args.fid_dims)}\n")

    logger.log(f"saved per-sample metrics to: {per_sample_csv}")
    logger.log(f"saved summary to: {out_dir / 'summary.json'}")
    logger.log(f"saved summary to: {out_dir / 'summary.txt'}")
    if bool(args.save_npy):
        logger.log(f"saved npy outputs to: {npy_dir}")


if __name__ == "__main__":
    main()
