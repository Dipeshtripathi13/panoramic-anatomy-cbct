"""Model, masked multi-label loss and metrics."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


def build_model(n_channels: int, encoder: str = "resnet34", pretrained: bool = True) -> torch.nn.Module:
    """U-Net with one sigmoid output per channel (channels overlap, so not a softmax)."""
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=1, classes=n_channels)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def masked_loss(logits, target, weight, dice_weight: float = 1.0, eps: float = 1.0):
    """Per-pixel BCE + per-(image, channel) soft Dice, both restricted to weight > 0.
    Channel/image pairs with no valid pixel contribute nothing."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * weight).sum() / weight.sum().clamp(min=1.0)
    p = torch.sigmoid(logits) * weight
    t = target * weight
    inter = (p * t).sum((2, 3))
    denom = p.sum((2, 3)) + t.sum((2, 3))
    valid = (weight.sum((2, 3)) > 0) & (t.sum((2, 3)) > 0)    # Dice only where the structure is present
    dice = 1 - (2 * inter + eps) / (denom + eps)
    dice = (dice * valid).sum() / valid.sum().clamp(min=1)
    return bce + dice_weight * dice, {"bce": bce.item(), "dice_loss": dice.item()}


@torch.no_grad()
def dice_scores(logits, target, weight, thr: float = 0.5):
    """Dice per (image, channel) over valid pixels; NaN where the structure is absent in both
    prediction and target, or where the channel has no valid pixels."""
    pred = (torch.sigmoid(logits) > thr).float() * weight
    t = target * weight
    inter = (pred * t).sum((2, 3))
    denom = pred.sum((2, 3)) + t.sum((2, 3))
    d = 2 * inter / denom.clamp(min=1e-6)
    d[(denom == 0) | (weight.sum((2, 3)) == 0)] = float("nan")
    return d.cpu()
