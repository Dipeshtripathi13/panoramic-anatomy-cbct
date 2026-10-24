"""Panoramic simulator v0 over ToothFairy3 (volumes in LPS after tf3.fix_orientation).

Geometry (PX2Tooth-style, patient-specific): the dental arch is a 4th-order polynomial
j(i) fitted to the axial projection of the tooth labels, extended along its end
tangents, and resampled by arclength. Every panoramic column is the line through the
arch point C(s) along the in-plane normal n(s); every row is an axial slice k
(row 0 = most superior). Patient right is on the image left (radiological convention).

Renderers on that geometry:
  mpr    — PX2Tooth "ideal panoramic": mean HU over a slab |t| <= mpr_halfwidth_mm.
  simpx  — Beer–Lambert transmittance as in NeBLa/X-Splat:
           p = (1 - I0 * exp(-beta * sum_k clip(HU)_k * ds)) / p_max, K samples per ray.
Multi-label masks: for each channel, a pixel is on if any voxel of that channel lies on
its ray within the channel's trough half-width (overlapping structures stay overlapping).

v0 deliberately has no tomographic blur, ghosting, beam tilt or magnification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

TOOTH_IDS = [q * 10 + t for q in (1, 2, 3, 4) for t in range(1, 9)]
PULP_IDS = [t + 100 for t in TOOTH_IDS]

# channel name -> (label ids, trough half-width in mm; None = whole simpx ray)
CHANNELS = {
    "teeth": (TOOTH_IDS, 12.0),
    "pulp": (PULP_IDS, 12.0),
    "canal_L": ([3], 12.0),
    "canal_R": ([4], 12.0),
    "incisive_canal": ([103, 104], 12.0),
    "sinus": ([5, 6], 20.0),
    "crown": ([9], 12.0),
    "bridge": ([8], 12.0),
    "implant": ([10], 12.0),
    "mandible": ([1], 12.0),
    "maxilla": ([2], 12.0),
    "pharynx": ([7], None),
}


@dataclass
class SimConfig:
    voxel_mm: float = 0.3
    column_mm: float = 0.3            # arclength step between columns
    arch_poly_deg: int = 4
    arch_extend_mm: float = 15.0      # extension beyond the last teeth along the end tangent
    mpr_halfwidth_mm: float = 12.0
    simpx_K: int = 200
    simpx_ray_mult: float = 0.5       # ray length = mult * axial diagonal of the volume
    simpx_beta: float = 7.5e-7
    simpx_ds: float = 1.35
    simpx_p_max: float = 0.25
    simpx_I0: float = 1.0
    hu_clip: tuple = (0.0, 4000.0)
    chunk_cols: int = 48
    channels: dict = field(default_factory=lambda: dict(CHANNELS))


# ---------------------------------------------------------------- arch curve
MIN_TEETH_FOR_ARCH = 6


def alveolar_crest_k(lab: np.ndarray, voxel_mm: float = 0.3) -> float | None:
    """Height (k, LPS: superior = larger) of the anterior mandibular crest: top of the mandible within
    15 mm of its most anterior point. None if there is no mandible label."""
    ii, jj, kk = np.nonzero(lab == 1)
    if len(kk) == 0:
        return None
    front = jj <= jj.min() + 15.0 / voxel_mm
    return float(np.percentile(kk[front], 98))


def arch_points_2d(lab: np.ndarray, voxel_mm: float = 0.3) -> tuple[np.ndarray, np.ndarray, str]:
    """(i, j) pixels that define the arch: tooth/prosthesis labels; with < MIN_TEETH_FOR_ARCH teeth
    (edentulous or nearly), also the mandibular body in a 15 mm slab below the alveolar crest."""
    teeth3d = np.isin(lab, TOOTH_IDS + [8, 9, 10])
    n_teeth = int(np.isin(TOOTH_IDS, np.unique(lab)).sum())
    teeth = teeth3d.any(axis=2)
    source = "teeth"
    if n_teeth < MIN_TEETH_FOR_ARCH:
        crest = alveolar_crest_k(lab, voxel_mm)
        if crest is not None:
            k0 = max(0, int(crest - 15.0 / voxel_mm))
            body = (lab[:, :, k0: int(crest) + 1] == 1).any(axis=2)
            teeth = teeth | body
            source = f"mandible_slab+{n_teeth}_teeth"
    ii, jj = np.nonzero(teeth)
    return ii, jj, source


def fit_arch(lab: np.ndarray, cfg: SimConfig) -> dict:
    """Arch curve in voxel (i, j) coordinates, resampled by arclength.

    Polar fit: around a centre behind the incisors (mean i; 75% of the way back in j),
    take the median radius of tooth pixels per 2° of angle θ (θ = 0 anterior, ±90° lateral),
    smooth ρ(θ) with a polynomial in θ, and map back. Unlike a polynomial j(i), this
    follows the near-sagittal posterior segments.
    """
    ii, jj, source = arch_points_2d(lab, cfg.voxel_mm)
    if len(ii) < 100:
        raise ValueError(f"cannot fit arch: only {len(ii)} tooth/mandible pixels")
    ci, cj = ii.mean(), jj.min() + 0.75 * (jj.max() - jj.min())
    theta = np.arctan2(ii - ci, cj - jj)
    rho = np.hypot(ii - ci, jj - cj)
    edges = np.deg2rad(np.arange(-180, 181, 2))
    idx = np.digitize(theta, edges)
    tb, rb, wb = [], [], []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() >= 20:
            tb.append(np.median(theta[sel]))
            rb.append(np.median(rho[sel]))
            wb.append(sel.sum())
    tb, rb, wb = map(np.asarray, (tb, rb, wb))
    coef = np.polyfit(tb, rb, cfg.arch_poly_deg, w=np.sqrt(wb))
    th = np.linspace(tb.min(), tb.max(), 4000)
    r = np.polyval(coef, th)
    pts = np.stack([ci + r * np.sin(th), cj - r * np.cos(th)], 1)

    ext = cfg.arch_extend_mm / cfg.voxel_mm
    t0 = pts[0] - pts[5]
    t1 = pts[-1] - pts[-6]
    t0, t1 = t0 / np.linalg.norm(t0), t1 / np.linalg.norm(t1)
    lin = np.linspace(0, ext, 200)[1:, None]
    pts = np.concatenate([pts[0] + lin[::-1] * t0, pts, pts[-1] + lin * t1])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    step = cfg.column_mm / cfg.voxel_mm
    s_new = np.arange(0, s[-1], step)
    P = np.stack([np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])], 1)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.stack([T[:, 1], -T[:, 0]], 1)
    mid = len(P) // 2
    if N[mid, 1] > 0:  # make normals point buccally (anterior = smaller j at the incisors)
        N = -N
    if not (np.isfinite(P).all() and np.isfinite(N).all()) or len(P) < 50:
        raise ValueError(f"degenerate arch fit ({source}, {len(P)} points)")
    return {"points": P, "normals": N, "length_mm": s_new[-1] * cfg.voxel_mm, "coef": coef, "source": source}


# ---------------------------------------------------------------- sampling
def _sample(vol: np.ndarray, arch: dict, t_vox: np.ndarray, cols: slice, order: int) -> np.ndarray:
    """Sample vol at C(s)+t*n(s) for all k. Returns (ncols, nk, nt)."""
    P, N = arch["points"][cols], arch["normals"][cols]
    nk = vol.shape[2]
    ci = P[:, 0, None] + t_vox[None, :] * N[:, 0, None]           # (c, t)
    cj = P[:, 1, None] + t_vox[None, :] * N[:, 1, None]
    shape = (len(P), nk, len(t_vox))
    I = np.broadcast_to(ci[:, None, :], shape)
    J = np.broadcast_to(cj[:, None, :], shape)
    K = np.broadcast_to(np.arange(nk)[None, :, None], shape)
    return ndimage.map_coordinates(vol, [I, J, K], order=order, mode="constant", cval=0 if order == 0 else -1000)


def render(img: np.ndarray, lab: np.ndarray, cfg: SimConfig | None = None) -> dict:
    """Render one LPS volume. Images are (rows=k from superior, cols=s from patient right)."""
    cfg = cfg or SimConfig()
    arch = fit_arch(lab, cfg)
    ncol = len(arch["points"])
    vx = cfg.voxel_mm

    t_mpr = np.arange(-cfg.mpr_halfwidth_mm, cfg.mpr_halfwidth_mm + 1e-6, vx) / vx
    diag = np.hypot(img.shape[0], img.shape[1])
    half = 0.5 * cfg.simpx_ray_mult * diag
    t_sim = np.linspace(-half, half, cfg.simpx_K)
    lo, hi = cfg.hu_clip

    mpr = np.zeros((ncol, img.shape[2]), np.float32)
    simpx = np.zeros_like(mpr)
    masks = {name: np.zeros((ncol, img.shape[2]), bool) for name in cfg.channels}
    for c0 in range(0, ncol, cfg.chunk_cols):
        sl = slice(c0, min(c0 + cfg.chunk_cols, ncol))
        mpr[sl] = _sample(img, arch, t_mpr, sl, order=1).mean(-1)
        hu = np.clip(_sample(img, arch, t_sim, sl, order=1), lo, hi)
        line = cfg.simpx_beta * hu.sum(-1) * cfg.simpx_ds
        simpx[sl] = (1.0 - cfg.simpx_I0 * np.exp(-line)) / cfg.simpx_p_max
        labs = _sample(lab, arch, t_sim, sl, order=0)
        for name, (ids, hw) in cfg.channels.items():
            within = np.ones_like(t_sim, bool) if hw is None else np.abs(t_sim) * vx <= hw
            masks[name][sl] = np.isin(labs[..., within], ids).any(-1)

    # orient: rows = superior→inferior, cols = patient right→left (i increasing = Left)
    flip_cols = arch["points"][0, 0] > arch["points"][-1, 0]

    def orient(a):
        a = a.T[::-1]
        return a[:, ::-1] if flip_cols else a
    return {
        "mpr": orient(mpr), "simpx": orient(simpx),
        "masks": {k: orient(v) for k, v in masks.items()},
        "arch": arch, "ray_half_mm": half * vx,
    }

