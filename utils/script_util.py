import argparse

from model.unet import UNetModel

NUM_CLASSES = 1000

MODEL_ARG_HELP = {
    "image_size": "Input image size in pixels. The released Mayo models use 512.",
    "in_channels": "Number of input channels to the U-Net. RDDM uses Gaussian noise plus LDCT, so this is 2.",
    "num_channels": "Base channel width of the U-Net backbone.",
    "out_channels": "Number of output channels. RDDM predicts one residual channel.",
    "num_res_blocks": "Number of residual blocks at each U-Net resolution level.",
    "num_heads": "Number of attention heads when attention layers are enabled.",
    "num_heads_upsample": "Attention heads used in upsampling blocks. -1 reuses num_heads.",
    "num_head_channels": "Channels per attention head. -1 disables this override.",
    "attention_resolutions": "Comma-separated spatial resolutions where attention is applied. Empty string disables attention.",
    "channel_mult": "Comma-separated U-Net channel multipliers by resolution. Empty string uses the default for image_size.",
    "dropout": "Dropout probability inside residual blocks.",
    "dims": "Signal dimensionality. Use 2 for 2D CT slices.",
    "class_cond": "Enable class conditioning in the U-Net. The released LDCT denoising model uses false.",
    "use_checkpoint": "Enable gradient checkpointing to reduce memory at the cost of extra compute.",
    "use_scale_shift_norm": "Use scale-shift normalization inside residual blocks.",
    "resblock_updown": "Use residual blocks for upsampling/downsampling instead of standalone sampling layers.",
    "use_new_attention_order": "Use the newer attention QKV ordering from the underlying U-Net implementation.",
}


def create_model(
    image_size,
    in_channels,
    num_channels,
    out_channels,
    num_res_blocks,
    dims=2,
    channel_mult="",
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(float(ch_mult) for ch_mult in str(channel_mult).split(","))

    attention_ds = []
    if attention_resolutions:
        for res in str(attention_resolutions).split(","):
            attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        dims=dims,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def add_dict_to_argparser(parser, default_dict, help_dict=None):
    help_dict = help_dict or {}
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(
            f"--{k}",
            default=v,
            type=v_type,
            help=help_dict.get(k, "No description provided."),
        )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")

