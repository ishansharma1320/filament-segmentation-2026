"""Network, loss, device and TTA forward."""
from __future__ import annotations

import os

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch: str = "unet", encoder: str = "resnet34", pretrained: bool = True):
    cls = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "fpn": smp.FPN,
           "deeplab": smp.DeepLabV3Plus, "manet": smp.MAnet}[arch]
    return cls(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
               in_channels=1, classes=1)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0,
                  pos_weight: float = 1.0) -> torch.Tensor:
    pw = torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return bce + dice_weight * dice.mean()


@torch.no_grad()
def predict_probs(model, x: torch.Tensor, tta: bool = True) -> torch.Tensor:
    """x: (B,1,S,S) -> sigmoid probs (B,1,S,S), optional 4x flip TTA."""
    out = torch.sigmoid(model(x))
    if not tta:
        return out
    out = out + torch.sigmoid(model(x.flip(-1))).flip(-1)
    out = out + torch.sigmoid(model(x.flip(-2))).flip(-2)
    out = out + torch.sigmoid(model(x.flip(-1, -2))).flip(-1, -2)
    return out / 4


def save_checkpoint(path, model, **meta):
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch"], ck["encoder"], pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
