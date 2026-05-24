"""
Train the strict residual-space RDDM for LDCT-to-NDCT denoising.

This script enforces drifting attraction/repulsion in residual space:
  r = y - x
and learns q_theta(r|x) toward p_data(r|x).
"""

import argparse
import datetime
import os

import torch as th
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.mayo_ima import MayoIMADataset, augment_defaults
from rddm.trainer import RDDMTrainConfig, RDDMTrainer
from utils import dist_util, logger
from utils.script_util import MODEL_ARG_HELP, add_dict_to_argparser, args_to_dict, create_model


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


TRAIN_ARG_HELP = {
    "data_dir": "Root dataset directory containing train/ldct and train/ndct subfolders.",
    "split_train": "Dataset split used for training, usually 'train'.",
    "recursive": "Search LDCT/NDCT folders recursively for .IMA or .dcm files.",
    "hu_min": "Lower HU bound used to normalize CT values before training.",
    "hu_max": "Upper HU bound used to normalize CT values before training.",
    "ncpus": "Number of DataLoader worker processes.",
    "batch_size": "Training batch size per process/GPU.",
    "lr": "Initial learning rate for AdamW.",
    "lr_decay_mode": "Learning-rate schedule mode. Use 'step' for step decay or 'none' to keep lr fixed.",
    "lr_decay_step": "Number of optimizer steps between LR decays when lr_decay_mode='step'.",
    "lr_decay_gamma": "Multiplicative LR decay factor used at each lr_decay_step.",
    "weight_decay": "Weight decay coefficient for AdamW.",
    "max_steps": "Total number of training optimization steps.",
    "ema_rate": "Exponential moving average decay for model weights saved in checkpoints.",
    "log_interval": "Print and record training logs every N steps.",
    "save_interval": "Save a checkpoint every N steps.",
    "resume_checkpoint": "Path to a checkpoint file for resuming training. Empty string starts from scratch.",
    "checkpointdir": "Directory where training checkpoints model_*.pth are written.",
    "use_fp16": "Enable mixed-precision training when CUDA is available.",
    "max_norm": "Gradient clipping max norm. Use 'none' to disable clipping.",
    "temperatures": "Comma-separated drifting temperatures, e.g. '1.0,1.5' for RDDM-Fine.",
    "drift_scale": "Scale applied to the computed residual drifting vector before forming the target.",
    "lambda_drift": "Weight of the residual drifting loss term.",
    "lambda_l1": "Weight of the optional residual L1 reconstruction loss.",
    "feature_eps": "Small epsilon for feature/residual normalization inside the drift kernel.",
    "distance_norm": "Distance normalization used in the drift kernel: 'sqrt_dim' or 'none'.",
}

AUGMENT_ARG_HELP = {
    "xflip": "Probability multiplier for paired horizontal flipping during training.",
    "yflip": "Probability multiplier for paired vertical flipping during training.",
    "rotate_int": "Probability multiplier for paired 90-degree rotations during training.",
    "translate_int": "Probability multiplier for paired integer-pixel translations during training.",
    "scale": "Probability multiplier for paired isotropic scale augmentation.",
    "rotate_frac": "Probability multiplier for paired arbitrary-angle rotation augmentation.",
    "aniso": "Probability multiplier for paired anisotropic scaling augmentation.",
    "translate_frac": "Probability multiplier for paired fractional-pixel translation augmentation.",
    "brightness": "Probability multiplier for brightness augmentation. Usually disabled for CT.",
    "contrast": "Probability multiplier for contrast augmentation. Usually disabled for CT.",
    "lumaflip": "Probability multiplier for luminance flip augmentation. Usually disabled for CT.",
    "hue": "Probability multiplier for hue augmentation. Not used for single-channel CT by default.",
    "saturation": "Probability multiplier for saturation augmentation. Not used for single-channel CT by default.",
}


def create_argparser():
    defaults = dict(
        data_dir="/path/to/aapm_mayo",
        split_train="train",
        recursive=True,
        hu_min=-1024.0,
        hu_max=3072.0,
        ncpus=20,
        batch_size=24,
        lr=1e-4,
        lr_decay_mode="step",
        lr_decay_step=10000,
        lr_decay_gamma=0.5,
        weight_decay=0.0,
        max_steps=50000,
        ema_rate=0.999,
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        checkpointdir="checkpoints/rddm_balanced",
        use_fp16=False,
        max_norm=1.0,
        temperatures="0.2,1.0",
        drift_scale=1.0,
        lambda_drift=1.0,
        lambda_l1=0.0,
        feature_eps=1e-8,
        distance_norm="sqrt_dim",
    )
    defaults.update(model_defaults())
    defaults.update(augment_defaults())
    help_dict = {}
    help_dict.update(TRAIN_ARG_HELP)
    help_dict.update(MODEL_ARG_HELP)
    help_dict.update(AUGMENT_ARG_HELP)
    parser = argparse.ArgumentParser(
        description="Train strict residual-space RDDM for paired LDCT-to-NDCT denoising.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dict_to_argparser(parser, defaults, help_dict=help_dict)
    return parser


def _parse_float_tuple(raw, name: str):
    if isinstance(raw, str):
        vals = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    elif isinstance(raw, (tuple, list)):
        vals = tuple(float(x) for x in raw)
    else:
        vals = (float(raw),)
    if len(vals) == 0:
        raise ValueError(f"--{name} must contain at least one value.")
    return vals


def main():
    args = create_argparser().parse_args()
    if isinstance(args.max_norm, str):
        args.max_norm = (
            None if args.max_norm.lower() == "none" else float(args.max_norm)
        )
    args.temperatures = _parse_float_tuple(args.temperatures, "temperatures")
    dist_util.setup_dist()
    logdir = os.path.join(
        "./logs", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), "rddm"
    )
    logger.configure(dir=logdir)

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
    model.to(dist_util.dev())

    logger.log("creating IMA dataloader...")
    aug_kwargs = args_to_dict(args, augment_defaults().keys())
    dataset = MayoIMADataset(
        root=args.data_dir,
        split=args.split_train,
        use_cond=True,
        recursive=args.recursive,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        **aug_kwargs,
    )
    train_sampler = None
    if dist.get_world_size() > 1:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=False,
        )

    data = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.ncpus,
        pin_memory=th.cuda.is_available(),
        drop_last=True,
    )

    cfg = RDDMTrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay_mode=args.lr_decay_mode,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpointdir=args.checkpointdir,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        max_norm=args.max_norm,
        temperatures=args.temperatures,
        drift_scale=args.drift_scale,
        lambda_drift=args.lambda_drift,
        lambda_l1=args.lambda_l1,
        feature_eps=args.feature_eps,
        distance_norm=args.distance_norm,
    )

    trainer = RDDMTrainer(
        model=model,
        train_loader=data,
        train_sampler=train_sampler,
        cfg=cfg,
        extra_state=vars(args),
    )
    trainer.run()


if __name__ == "__main__":
    main()
