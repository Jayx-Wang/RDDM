import glob
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from utils import dist_util, logger
from .losses import drifting_objective

try:
    from torch.amp import GradScaler as TorchGradScaler
    from torch.amp import autocast as torch_autocast

    _HAS_TORCH_AMP = True
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore
    from torch.cuda.amp import autocast as torch_autocast  # type: ignore

    _HAS_TORCH_AMP = False


@dataclass
class RDDMTrainConfig:
    batch_size: int
    lr: float
    lr_decay_mode: str
    lr_decay_step: int
    lr_decay_gamma: float
    weight_decay: float
    max_steps: int
    ema_rate: float
    log_interval: int
    save_interval: int
    checkpointdir: str
    resume_checkpoint: str
    use_fp16: bool
    max_norm: Optional[float]
    temperatures: Tuple[float, ...]
    drift_scale: float
    lambda_drift: float
    lambda_l1: float
    feature_eps: float
    distance_norm: str


def _find_latest_checkpoint(checkpointdir: str) -> Optional[str]:
    ckpts = glob.glob(os.path.join(checkpointdir, "model_*.pth"))
    if len(ckpts) == 0:
        return None
    ckpts.sort()
    return ckpts[-1]


class RDDMTrainer:
    """Strict residual-space RDDM trainer for paired LDCT/NDCT slices."""

    def __init__(
        self,
        *,
        model,
        train_loader: Iterable,
        train_sampler,
        cfg: RDDMTrainConfig,
        extra_state: Optional[Dict] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.cfg = cfg
        self.extra_state = extra_state or {}

        self.device = dist_util.dev()
        self.step = 0
        self.epoch = 0
        self.global_batch = cfg.batch_size * dist.get_world_size()
        self.data_iter = iter(self.train_loader)

        self.opt = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        mode = str(cfg.lr_decay_mode).lower()
        if mode == "step":
            if int(cfg.lr_decay_step) <= 0:
                raise ValueError(f"lr_decay_step must be > 0, got {cfg.lr_decay_step}")
            if not (0.0 < float(cfg.lr_decay_gamma) < 1.0):
                raise ValueError(f"lr_decay_gamma must be in (0, 1), got {cfg.lr_decay_gamma}")
            self.scheduler = StepLR(
                self.opt,
                step_size=int(cfg.lr_decay_step),
                gamma=float(cfg.lr_decay_gamma),
            )
        elif mode in {"none", ""}:
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported lr_decay_mode: {cfg.lr_decay_mode}")

        if _HAS_TORCH_AMP:
            self.scaler = TorchGradScaler(
                "cuda", enabled=(cfg.use_fp16 and th.cuda.is_available())
            )
        else:
            self.scaler = TorchGradScaler(enabled=cfg.use_fp16)

        self.ema_params = [p.detach().clone() for p in self.model.parameters()]
        logger.log("strict residual-space RDDM: enabled")
        logger.log(f"temperatures: {tuple(float(t) for t in cfg.temperatures)}")
        logger.log(f"lambda_l1: {float(cfg.lambda_l1)}")

        self._load_checkpoint_if_needed()

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.use_ddp = False
            self.ddp_model = self.model

    def _barrier(self):
        if not dist.is_available() or not dist.is_initialized():
            return
        if th.cuda.is_available() and dist.get_backend() == "nccl":
            dist.barrier(device_ids=[th.cuda.current_device()])
        else:
            dist.barrier()

    def _load_checkpoint_if_needed(self):
        if dist.get_rank() == 0 and not os.path.exists(self.cfg.checkpointdir):
            os.makedirs(self.cfg.checkpointdir, exist_ok=True)
        self._barrier()

        ckpt_path = str(self.cfg.resume_checkpoint).strip()
        if ckpt_path == "":
            logger.log("resume_checkpoint is empty; start training from scratch.")
            return

        logger.log(f"loading checkpoint: {ckpt_path}")
        state = th.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state["model"])

        if "optimizer" in state:
            self.opt.load_state_dict(state["optimizer"])
        self.step = int(state.get("step", 0))
        if self.scheduler is not None:
            if "scheduler" in state and state["scheduler"] is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            else:
                self.scheduler.step(self.step)

        if "ema" in state:
            for ema_src, ema_tgt in zip(state["ema"], self.ema_params):
                ema_tgt.data.copy_(ema_src.to(ema_tgt.device, dtype=ema_tgt.dtype))

    def _next_batch(self):
        while True:
            try:
                return next(self.data_iter)
            except StopIteration:
                self.epoch += 1
                if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
                    self.train_sampler.set_epoch(self.epoch)
                self.data_iter = iter(self.train_loader)

    def _save_checkpoint(self):
        if dist.get_rank() == 0:
            ckpt_name = f"model_{self.step:06d}.pth"
            ckpt_path = os.path.join(self.cfg.checkpointdir, ckpt_name)
            logger.log(f"saving checkpoint: {ckpt_path}")
            th.save(
                {
                    "model": self.model.state_dict(),
                    "ema": [p.detach().cpu() for p in self.ema_params],
                    "optimizer": self.opt.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "step": self.step,
                    "extra_state": self.extra_state,
                },
                ckpt_path,
            )
        self._barrier()

    def _update_ema(self):
        with th.no_grad():
            for tgt, src in zip(self.ema_params, self.model.parameters()):
                tgt.mul_(self.cfg.ema_rate).add_(src, alpha=1.0 - self.cfg.ema_rate)

    def _run_step(self, batch, cond):
        self.opt.zero_grad(set_to_none=True)

        ndct = batch.to(self.device)
        ldct = cond["x_prime"].to(self.device)
        if ndct.shape[0] < 2:
            raise ValueError(
                "RDDM drifting loss requires local batch_size >= 2. "
                "Increase --batch_size or use fewer distributed processes."
            )

        noise = th.randn_like(ndct)

        def make_amp_ctx():
            if _HAS_TORCH_AMP:
                return torch_autocast(
                    "cuda", enabled=(self.cfg.use_fp16 and th.cuda.is_available())
                )
            if self.cfg.use_fp16:
                return torch_autocast(enabled=True)
            return nullcontext()

        with make_amp_ctx():
            pred = self.ddp_model(noise, x_prime=ldct)
            residual_gen = th.tanh(pred)
            recon = (ldct + residual_gen).clamp(-1.0, 1.0)

        residual_pos = ndct - ldct
        loss_dict = drifting_objective(
            residual_gen,
            residual_pos,
            temperatures=self.cfg.temperatures,
            drift_scale=self.cfg.drift_scale,
            lambda_drift=self.cfg.lambda_drift,
            lambda_l1=self.cfg.lambda_l1,
            feature_eps=self.cfg.feature_eps,
            distance_norm=self.cfg.distance_norm,
        )

        self.scaler.scale(loss_dict["loss"]).backward()
        if self.cfg.max_norm is not None:
            self.scaler.unscale_(self.opt)
            th.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_norm)
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self._update_ema()

        log_dict = {k: v.detach().float().item() for k, v in loss_dict.items()}
        log_dict["loss_recon_l1"] = float(F.l1_loss(recon, ndct).detach().float().item())
        return log_dict

    def run(self):
        logger.log("start RDDM training...")
        pbar = None
        if dist.get_rank() == 0:
            pbar = tqdm(
                total=self.cfg.max_steps,
                initial=self.step,
                dynamic_ncols=True,
                desc="rddm-train",
            )

        while self.step < self.cfg.max_steps:
            batch, cond = self._next_batch()
            self.step += 1
            loss_dict = self._run_step(batch, cond)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss_dict['loss']:.4e}",
                    drift=f"{loss_dict['loss_drift']:.4e}",
                    l1=f"{loss_dict['loss_l1']:.4e}",
                )

            logger.logkv("step", self.step)
            logger.logkv("samples", self.step * self.global_batch)
            logger.logkv("lr", self.opt.param_groups[0]["lr"])
            for key, value in loss_dict.items():
                logger.logkv_mean(key, value)

            if self.step % self.cfg.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.cfg.save_interval == 0:
                self._save_checkpoint()

        if self.step % self.cfg.save_interval != 0:
            self._save_checkpoint()
        if pbar is not None:
            pbar.close()
