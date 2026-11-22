"""CycleGAN baseline (NeBLa-style): unpaired translation synthetic panoramic → real DENTEX appearance.

Standard architecture (Zhu et al. 2017): ResNet generator (2 down, n residual blocks, 2 up, instance norm), 70×70
PatchGAN discriminator, least-squares GAN loss, cycle λ = 10, identity 0.5λ, image pool of 50. Images are 1-channel in
[0, 1]; the networks work in [-1, 1]. Fully convolutional: trained on random square crops, applied to whole panoramics.
The segmentation baseline then trains on G_AB(synthetic) with the original synthetic masks (scripts/train.py,
`synth_translate`).
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from .. import io as dio
from .data import REAL_CROP, _resize, normalise_real


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
                                  nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    def __init__(self, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(1, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        c = ngf
        for _ in range(2):
            layers += [nn.Conv2d(c, 2 * c, 3, 2, 1), nn.InstanceNorm2d(2 * c), nn.ReLU(True)]
            c *= 2
        layers += [ResBlock(c) for _ in range(n_blocks)]
        for _ in range(2):
            layers += [nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(c, 1, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x in [0, 1] → translated image in [0, 1]; H and W must be multiples of 4."""
        return (self.net(x * 2 - 1) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64):
        super().__init__()
        layers, c = [nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)], ndf
        for stride in (2, 2, 1):
            layers += [nn.Conv2d(c, 2 * c, 4, stride, 1), nn.InstanceNorm2d(2 * c), nn.LeakyReLU(0.2, True)]
            c *= 2
        layers += [nn.Conv2d(c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x * 2 - 1)


class ImagePool:
    """History of generated images for the discriminator update (Shrivastava et al. 2017)."""

    def __init__(self, size: int = 50):
        self.size, self.items = size, []

    def query(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for im in x.detach():
            if len(self.items) < self.size:
                self.items.append(im)
                out.append(im)
            elif random.random() < 0.5:
                k = random.randrange(self.size)
                out.append(self.items[k])
                self.items[k] = im
            else:
                out.append(im)
        return torch.stack(out)


class DentexImages(Dataset):
    """Real DENTEX images without labels ('<zip>:<subset>/<file>' names), same fixed crop/normalisation as
    DentexDataset, resized to the panoramic canvas."""

    def __init__(self, names: list[str], size=(256, 640), crop=REAL_CROP):
        self.names, self.size, self.crop = list(names), tuple(size), crop

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        zip_key, rest = self.names[i].split(":", 1)
        subset, fname = rest.split("/")
        zf = dio.open_zip("dentex", f"{zip_key}_zip")
        root = "training_data" if zip_key == "train" else "validation_data"
        img = np.array(Image.open(zf.open(f"{root}/{subset}/xrays/{fname}")).convert("L"))
        H, W = img.shape
        x0, y0, x1, y1 = int(self.crop[0] * W), int(self.crop[1] * H), int(self.crop[2] * W), int(self.crop[3] * H)
        img = torch.from_numpy(normalise_real(img[y0:y1, x0:x1]))[None]
        return _resize(img, self.size, "bilinear"), self.names[i]


def load_generator(path, device=None, key: str = "G_AB") -> Generator:
    """key G_AB = synthetic → real (training-time baseline), G_BA = real → synthetic (NeBLa-style test-time)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    g = Generator(**ck.get("arch", {}))
    g.load_state_dict(ck[key])
    g.eval().requires_grad_(False)
    return g.to(device) if device is not None else g


class Translated(nn.Module):
    """A segmentation model applied after a frozen translator: model(G(x).clamp(0, 1))."""

    def __init__(self, generator: nn.Module, model: nn.Module):
        super().__init__()
        self.generator, self.model = generator, model

    def forward(self, x):
        return self.model(self.generator(x).clamp(0, 1))
