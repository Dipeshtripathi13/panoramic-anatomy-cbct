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
    arch_smooth_deg: float = 6.0      # Gaussian smoothing of the polar arch radius ρ(θ)
    arch_extend_mm: float = 10.0      # straight extension beyond the arch data along the end tangent
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
def alveolar_crest_k(lab: np.ndarray, voxel_mm: float = 0.3) -> float | None:
    """Height (k, LPS: superior = larger) of the anterior mandibular crest: top of the mandible within
    15 mm of its most anterior point. None if there is no mandible label."""
    ii, jj, kk = np.nonzero(lab == 1)
    if len(kk) == 0:
        return None
    front = jj <= jj.min() + 15.0 / voxel_mm
    return float(np.percentile(kk[front], 98))


def mandible_slab_2d(lab: np.ndarray, voxel_mm: float = 0.3, below_mm: float = 20.0, above_mm: float = 10.0):
    """Axial projection of the mandible in a slab around the anterior alveolar crest
    (body + lower rami, where the canal runs). None if there is no mandible label."""
    crest = alveolar_crest_k(lab, voxel_mm)
    if crest is None:
        return None
    k0 = max(0, int(crest - below_mm / voxel_mm))
    k1 = min(lab.shape[2], int(crest + above_mm / voxel_mm) + 1)
    m = (lab[:, :, k0:k1] == 1).any(axis=2)
    return m if m.sum() >= 100 else None


def fit_arch(lab: np.ndarray, cfg: SimConfig) -> dict:
    """Arch curve (focal-trough centre line) in voxel (i, j) coordinates, resampled by arclength.

    Polar fit around a centre inside the mandibular U: for each 2° direction θ (0 = anterior), the
    radius is the middle of the combined 5th–95th-percentile radial range of the TEETH and the
    mandibular CANAL in that direction (so the trough holds both labelled structures); directions
    with neither use the median of the MANDIBULAR BODY (slab around the alveolar crest), into the rami.
    History: a teeth-only fit + straight extension missed the posterior canal in ~125/532 volumes;
    a teeth-else-mandible fit fixed the canal but lost up to 29% of the teeth in 48 volumes.
    ρ(θ) is smoothed with a Gaussian over θ (σ = cfg.arch_smooth_deg).
    """
    teeth = np.isin(lab, TOOTH_IDS + [8, 9, 10]).any(axis=2)
    mand = mandible_slab_2d(lab, cfg.voxel_mm)
    ref = mand if mand is not None else teeth
    ri, rj = np.nonzero(ref)
    if len(ri) < 100:
        raise ValueError(f"cannot fit arch: only {len(ri)} mandible/tooth pixels")
    ci, cj = ri.mean(), rj.min() + 0.6 * (rj.max() - rj.min())

    def polar(mask):
        ii, jj = np.nonzero(mask)
        return np.arctan2(ii - ci, cj - jj), np.hypot(ii - ci, jj - cj)

    edges = np.deg2rad(np.arange(-180, 181, 2))
    canal = np.isin(lab, (3, 4)).any(axis=2)
    th_t, rho_t = polar(teeth)
    th_c, rho_c = polar(canal)
    th_m, rho_m = polar(mand) if mand is not None else (np.array([]), np.array([]))
    bt, bc, bm = np.digitize(th_t, edges), np.digitize(th_c, edges), np.digitize(th_m, edges)
    tb, rb, nteeth_bins = [], [], 0
    for b in range(1, len(edges)):
        st, sc, sm = bt == b, bc == b, bm == b
        parts = [rho_t[st]] if st.sum() >= 20 else []
        if sc.sum() >= 10:
            parts.append(rho_c[sc])
        if parts:
            # centre the trough on the labelled structures in this direction (teeth and canal):
            # middle of their combined 5th–95th percentile radial range
            lo = min(np.percentile(x, 5) for x in parts)
            hi = max(np.percentile(x, 95) for x in parts)
            ths = np.concatenate([th_t[st], th_c[sc]]) if st.sum() >= 20 else th_c[sc]
            tb.append(np.median(ths))
            rb.append(0.5 * (lo + hi))
            nteeth_bins += st.sum() >= 20
        elif sm.sum() >= 30:
            tb.append(np.median(th_m[sm]))
            rb.append(np.median(rho_m[sm]))
    if len(tb) < 10:
        raise ValueError(f"cannot fit arch: only {len(tb)} angular bins")
    tb, rb = np.asarray(tb), np.asarray(rb)
    order = np.argsort(tb)
    tb, rb = tb[order], rb[order]
    th = np.linspace(tb[0], tb[-1], 4000)
    r = ndimage.gaussian_filter1d(np.interp(th, tb, rb), np.deg2rad(cfg.arch_smooth_deg) / (th[1] - th[0]), mode="nearest")
    pts = np.stack([ci + r * np.sin(th), cj - r * np.cos(th)], 1)
    source = "teeth" if nteeth_bins == len(tb) else ("mandible" if nteeth_bins == 0 else "teeth+mandible")

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
    return {"points": P, "normals": N, "length_mm": s_new[-1] * cfg.voxel_mm, "source": source}


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



def arch_distance_map(arch: dict, shape_ij: tuple) -> np.ndarray:
    """In-plane distance (voxels) from every (i, j) to the arch curve."""
    curve = np.zeros(shape_ij, bool)
    p = np.round(arch["points"]).astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 0] < shape_ij[0]) & (p[:, 1] >= 0) & (p[:, 1] < shape_ij[1])
    curve[p[ok, 0], p[ok, 1]] = True
    return ndimage.distance_transform_edt(~curve)
