"""Panoramic-specific domain randomisation (contribution 2).

Every factor has a nominal value (what the un-randomised simulator uses) and a sampler.
An ablation fixes one factor at nominal while the others stay random. Factors act at three
levels, which decides what must be re-rendered:
  volume   — changes the attenuation volume before rendering      (needs a new render)
  geometry — changes the OPG configuration or the arch            (needs a new render)
  image    — acts on the rendered mean transmission                (cheap, done on the fly)
Brief factor → implementation:
  trough_width  r_min/r_max scale (rotation-centre distance; larger → wider sharp layer)
  blur_outside  slit width (wider slit → stronger blur away from the trough)
  arch_jitter   patient positioning: arch shifted AP/lateral, rotated, scaled; beam tilt
  ghosts        weight of attenuation > 50 mm behind the trough (far jaw, ramus)
  spine         scale of the cervical-spine excess attenuation
  metal         scale of metal excess attenuation (crown/bridge/implant labels only: enamel reaches
                the scanner HU ceiling, so an HU threshold would also hit enamel)
  contrast      μ_water (beam energy) and bone/tooth contrast in the HU→μ map
  noise         photon count (Poisson) + electronic noise + detector PSF
  processing    vendor post-processing: background subtraction, unsharp mask, gamma
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage

from .opg import OPGConfig, process

U = lambda lo, hi: (lambda rng: float(rng.uniform(lo, hi)))
LOGU = lambda lo, hi: (lambda rng: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))))

FACTORS = {
    # name: (level, {param: (nominal, sampler)})
    "trough_width": ("geometry", {"r_scale": (1.0, LOGU(0.7, 1.6))}),
    "blur_outside": ("geometry", {"slit_mm": (6.0, U(3.0, 10.0))}),
    "arch_jitter": ("geometry", {"shift_ap_mm": (0.0, U(-4.0, 4.0)), "shift_lat_mm": (0.0, U(-2.0, 2.0)),
                                 "rot_deg": (0.0, U(-4.0, 4.0)), "scale": (1.0, U(0.95, 1.05)),
                                 "tilt_deg": (5.0, U(0.0, 10.0))}),
    "ghosts": ("geometry", {"far_weight": (1.0, U(0.4, 1.3))}),
    "spine": ("volume", {"spine_scale": (1.0, U(0.3, 1.6))}),
    "metal": ("volume", {"metal_scale": (1.0, LOGU(1.0, 3.0))}),
    "contrast": ("volume", {"mu_water": (0.02, U(0.016, 0.026)), "bone_contrast": (1.0, U(0.75, 1.3))}),
    "noise": ("image", {"photons": (np.inf, LOGU(2e3, 5e4)), "read_noise": (0.0, U(0.0, 0.01)),
                        "psf_mm": (0.0, U(0.0, 0.4))}),
    "processing": ("image", {"bg_sigma_mm": (12.0, U(6.0, 25.0)), "bg_amount": (0.85, U(0.6, 0.95)),
                             "unsharp": (0.0, U(0.0, 1.0)), "gamma": (1.0, LOGU(0.7, 1.4))}),
}


def sample(rng: np.random.Generator, enabled: set[str] | None = None) -> dict:
    """Parameter dict: enabled factors are sampled, the rest are nominal.

    Every sampler is always drawn (in a fixed order) so that, for the same rng seed, an ablation
    sample differs from the fully randomised one ONLY in the disabled factor (paired ablation)."""
    enabled = set(FACTORS) if enabled is None else set(enabled)
    unknown = enabled - set(FACTORS)
    assert not unknown, f"unknown factors {unknown}"
    out = {}
    for name, (_, params) in FACTORS.items():
        for p, (nominal, sampler) in params.items():
            v = sampler(rng)
            out[p] = v if name in enabled else nominal
    return out


def nominal() -> dict:
    return sample(np.random.default_rng(0), enabled=set())


# ------------------------------------------------------------------ volume level
def spine_mask(img: np.ndarray, lab: np.ndarray, voxel_mm: float = 0.3, half_width_mm: float = 25.0) -> np.ndarray:
    """Bone voxels posterior to the airway (pharynx, label 7) and within ±half_width of its midline,
    excluding labelled structures — i.e. the cervical spine, not skull base/mastoid."""
    ph = lab == 7
    if not ph.any():
        return np.zeros(lab.shape, bool)
    jmax = np.full(lab.shape[2], np.inf)
    ks = np.nonzero(ph.any((0, 1)))[0]
    for k in ks:
        jmax[k] = np.nonzero(ph[:, :, k].any(0))[0].max()
    jmax[: ks.min()] = jmax[ks.min()]                   # extend below/above the labelled airway
    jmax[ks.max() + 1:] = jmax[ks.max()]
    j = np.arange(lab.shape[1])[None, :, None]
    ic = np.nonzero(ph)[0].mean()
    i = np.arange(lab.shape[0])[:, None, None]
    region = (j > jmax[None, None, :] + 2.0 / voxel_mm) & (np.abs(i - ic) <= half_width_mm / voxel_mm) & (lab == 0)
    cortex = ndimage.binary_opening((img > 500) & region, iterations=1)
    # cancellous bone inside vertebrae is < 500 HU in CBCT: close the cortical shell and fill it per slice
    closed = ndimage.binary_closing(cortex, iterations=3)
    filled = np.stack([ndimage.binary_fill_holes(closed[:, :, k]) for k in range(closed.shape[2])], 2)
    return filled & region & (img > 0)


def attenuation(img: np.ndarray, lab: np.ndarray, p: dict, spine: np.ndarray | None = None) -> np.ndarray:
    """μ (per mm) volume with contrast, spine and metal factors applied."""
    excess = np.clip(img, -1000, None) / 1000.0 * p["bone_contrast"]       # HU/1000, scaled
    mu = p["mu_water"] * (1.0 + excess)
    if p["spine_scale"] != 1.0 and spine is not None and spine.any():
        mu[spine] = p["mu_water"] * (1.0 + excess[spine] * p["spine_scale"])
    if p["metal_scale"] != 1.0:
        metal = np.isin(lab, (8, 9, 10))
        mu[metal] = p["mu_water"] * (1.0 + excess[metal] * p["metal_scale"])
    return np.clip(mu, 0, None).astype(np.float32)


# ------------------------------------------------------------------ geometry level
def opg_config(p: dict, base: OPGConfig | None = None) -> OPGConfig:
    base = base or OPGConfig()
    return dataclasses.replace(base, r_min_mm=base.r_min_mm * p["r_scale"], r_max_mm=base.r_max_mm * p["r_scale"],
                               slit_mm=p["slit_mm"], tilt_deg=p["tilt_deg"], far_weight=p["far_weight"],
                               mu_water_per_mm=p["mu_water"])


def jitter_arch(arch: dict, p: dict, voxel_mm: float = 0.3) -> dict:
    """Rigid+scale perturbation of the focal-trough curve relative to the anatomy (positioning error)."""
    P = arch["points"]
    c = P.mean(0)
    th = np.deg2rad(p["rot_deg"])
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = (P - c) @ Rm.T * p["scale"] + c
    Q = Q + np.array([p["shift_lat_mm"], p["shift_ap_mm"]]) / voxel_mm   # i lateral, j AP (+ = posterior)
    # re-sample by arclength so one frame stays one column step
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    step = np.median(np.linalg.norm(np.diff(P, axis=0), axis=1))
    sn = np.arange(0, s[-1], step)
    Q = np.stack([np.interp(sn, s, Q[:, 0]), np.interp(sn, s, Q[:, 1])], 1)
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.stack([T[:, 1], -T[:, 0]], 1)
    if N[len(N) // 2, 1] > 0:
        N = -N
    return {**arch, "points": Q, "normals": N}


# ------------------------------------------------------------------ image level
def image_from_transmission(meanT: np.ndarray, p: dict, px_mm: float, rng: np.random.Generator) -> np.ndarray:
    """Noise, detector PSF and vendor processing on the mean transmission → image in [0, 1]."""
    T = meanT.astype(np.float64)
    if np.isfinite(p["photons"]):
        T = rng.poisson(T * p["photons"]) / p["photons"]
    if p["read_noise"] > 0:
        T = T + rng.normal(0, p["read_noise"] * T.mean(), T.shape)
    if p["psf_mm"] > 0:
        T = ndimage.gaussian_filter(T, p["psf_mm"] / px_mm)
    img = -np.log(np.clip(T, 1e-5, 1))
    if p["unsharp"] > 0:
        img = img + p["unsharp"] * (img - ndimage.gaussian_filter(img, 1.0 / px_mm))
    return process(img, px_mm, sigma_mm=p["bg_sigma_mm"], amount=p["bg_amount"], gamma=p["gamma"]).astype(np.float32)
