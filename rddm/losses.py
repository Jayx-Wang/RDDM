import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


def _to_flat_features(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1).to(torch.float32)


def _normalize_feature_space(
    gen_feat: torch.Tensor,
    pos_feat: torch.Tensor,
    neg_feat: torch.Tensor,
    eps: float,
    distance_norm: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen_feat = F.normalize(gen_feat, p=2, dim=1, eps=eps)
    pos_feat = F.normalize(pos_feat, p=2, dim=1, eps=eps)
    neg_feat = F.normalize(neg_feat, p=2, dim=1, eps=eps)

    all_feat = torch.cat([gen_feat, pos_feat, neg_feat], dim=0)
    scale = all_feat.std(dim=0, unbiased=False).clamp_min(eps)
    gen_feat = gen_feat / scale
    pos_feat = pos_feat / scale
    neg_feat = neg_feat / scale

    if distance_norm == "sqrt_dim":
        norm_factor = math.sqrt(gen_feat.shape[1])
        gen_feat = gen_feat / norm_factor
        pos_feat = pos_feat / norm_factor
        neg_feat = neg_feat / norm_factor
    elif distance_norm != "none":
        raise ValueError(f"Unsupported distance_norm: {distance_norm}")

    return gen_feat, pos_feat, neg_feat


def compute_drift(
    gen: torch.Tensor,
    pos: torch.Tensor,
    neg: Optional[torch.Tensor] = None,
    temp: float = 0.05,
    feature_eps: float = 1e-8,
    distance_norm: str = "sqrt_dim",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the residual-space drifting vector for generated samples."""
    if gen.shape != pos.shape:
        raise ValueError(
            f"gen and pos must have the same shape, got {gen.shape} and {pos.shape}."
        )
    if gen.ndim < 2:
        raise ValueError("Expected batch tensors with shape [B, ...].")
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}.")

    gen_flat = _to_flat_features(gen)
    pos_flat = _to_flat_features(pos)
    neg_flat = gen_flat if neg is None else _to_flat_features(neg)
    n = neg_flat.shape[0]

    gen_feat, pos_feat, neg_feat = _normalize_feature_space(
        gen_flat, pos_flat, neg_flat, eps=feature_eps, distance_norm=distance_norm
    )

    feat_targets = torch.cat([neg_feat, pos_feat], dim=0)
    sample_targets = torch.cat([neg_flat, pos_flat], dim=0)
    dist = torch.cdist(gen_feat, feat_targets)
    kernel = (-dist / temp).exp()

    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalized_kernel = kernel / normalizer.clamp_min(eps).sqrt()
    neg_kernel = normalized_kernel[:, :n]
    pos_kernel = normalized_kernel[:, n:]

    pos_coeff = pos_kernel * neg_kernel.sum(dim=-1, keepdim=True)
    neg_coeff = neg_kernel * pos_kernel.sum(dim=-1, keepdim=True)
    pos_v = pos_coeff @ sample_targets[n:]
    neg_v = neg_coeff @ sample_targets[:n]
    return (pos_v - neg_v).to(gen.dtype).view_as(gen)


def drifting_objective(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temperatures: Iterable[float] = (0.2, 1.0),
    drift_scale: float = 1.0,
    lambda_drift: float = 1.0,
    lambda_l1: float = 0.0,
    feature_eps: float = 1e-8,
    distance_norm: str = "sqrt_dim",
) -> Dict[str, torch.Tensor]:
    """Strict residual-space RDDM objective."""
    temps = tuple(float(t) for t in temperatures)
    if len(temps) == 0:
        raise ValueError("temperatures must be non-empty.")

    drift_loss_terms = []
    drift_norm_terms = []
    for temp in temps:
        with torch.no_grad():
            drift = compute_drift(
                gen,
                pos,
                neg=gen,
                temp=temp,
                feature_eps=feature_eps,
                distance_norm=distance_norm,
            )
            target = (gen + drift_scale * drift).detach()
        drift_loss_terms.append(F.mse_loss(gen, target))
        drift_norm_terms.append(drift.abs().mean())

    loss_drift = torch.stack(drift_loss_terms).sum()
    loss_l1 = F.l1_loss(gen, pos)
    loss = lambda_drift * loss_drift + lambda_l1 * loss_l1
    drift_norm = torch.stack(drift_norm_terms).mean()
    return {
        "loss": loss,
        "loss_drift": loss_drift,
        "loss_l1": loss_l1,
        "drift_norm": drift_norm,
    }
