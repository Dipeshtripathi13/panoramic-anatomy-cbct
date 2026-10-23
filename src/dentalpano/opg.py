"""Rotational panoramic (orthopantomograph) simulator with time-delay integration (TDI).

Geometry. Time is parametrised by arclength s along the patient's arch curve A(s) (the
focal-trough centre line), one step per `column_mm`. At step s the central ray passes through
A(s) with in-plane direction u(φ(s)) (pointing buccally, from source to detector), tilted
upward by `tilt_deg`. Consecutive central rays intersect at the instantaneous rotation centre;
we choose its distance r(s) behind the trough (lingually), which fixes the beam angle by the
envelope condition
        dφ/ds = (T · u⊥) / r(s)            (T = arch tangent, u⊥ = u rotated +90°)
integrated from the midline, where the beam is normal to the arch. Beam modes:
  "curvature" (default): r(s) = arch radius of curvature (smoothed), clipped to [r_min, r_max],
      plus a weak pull `k_normal` toward the arch normal. The beam stays near-perpendicular
      to the teeth like real machines; the rotation centre is close behind the anterior
      teeth (narrow trough) and far behind the posterior ones (wide trough).
  "rotating": fixed r profile r_front → r_back. Simple, but beams drift 20–26° oblique
      posteriorly, which smears teeth (bead phantom, LOG.md).
  "normal": exact arch normals (r = radius of curvature, unclipped; nearly a translation in
      straight posterior segments, so almost no depth blur).
Source = A − src_trough·u; detector centre = source + sid·u; slit of `slit_mm` with horizontal
axis e ⟂ u (oriented along T) and vertical axis w ⟂ u.
TDI: a trough point's projection moves at M·(T·e) = M·cos δ per unit s, where δ is the beam
obliquity, so the detector advances by X'(s) = M·cos δ(s). Trough points stay fixed and sharp;
other depths drift and blur; structures on the far side smear into ghosts, which the tilt
lifts upward.
Image formation: transmission T = exp(−∫μ ds) is averaged over the exposure (the detector
integrates photons) and shown as −log(mean T). μ = μ_water·(1 + HU/1000) clipped at 0
(physical), or β·clip(HU, 0, 4000) (X-Splat SimPX constants).
Masks: per channel, the exposure fraction in which some voxel of the channel lies on the ray
within ±half-width of the ray's focal point (depth window along the ray around the trough
crossing, so far-side structures that the ray also crosses are excluded); thresholded at 0.5.
Overlaps are kept.
Display: `process()` mimics vendor post-processing (large-scale background subtraction +
gamma); the raw −log image is kept so processing can be randomised later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .simulator import CHANNELS, PULP_IDS, TOOTH_IDS, SimConfig, fit_arch

OPG_CHANNELS = dict(CHANNELS, teeth=(TOOTH_IDS + PULP_IDS, 12.0))  # a tooth includes its pulp


@dataclass
class OPGConfig:
    voxel_mm: float = 0.3
    column_mm: float = 0.3          # arch travel per time step
    arch_extend_mm: float = 25.0
    beam: str = "curvature"         # "curvature" | "rotating" | "normal"
    r_min_mm: float = 15.0          # curvature mode: clip range for the rotation-centre distance
    r_max_mm: float = 80.0
    curvature_smooth_mm: float = 5.0
    k_normal_per_mm: float = 0.3    # curvature mode: pull of the beam angle toward the arch normal
    r_front_mm: float = 20.0        # rotation-centre distance behind the trough at the midline ...
    r_back_mm: float = 45.0         # ... and in the posterior segments
    r_transition_mm: tuple = (10.0, 35.0)  # |s| from midline over which r goes front → back
    sid_mm: float = 500.0           # source–image distance
    src_trough_mm: float = 400.0    # source–focal-trough distance → M = 1.25
    slit_mm: float = 6.0            # slit width at the detector
    tilt_deg: float = 5.0           # central ray rises toward the detector
    depth_before_mm: float = 140.0  # ray sampled from (trough − before) ...
    depth_after_mm: float = 40.0    # ... to (trough + after), along the ray
    step_mm: float = 0.6
    mode: str = "physical"          # "physical" | "simpx"
    mu_water_per_mm: float = 0.02
    simpx_beta: float = 7.5e-7
    simpx_ds_per_mm: float = 3.1    # matches v0 SimPX (ds = 1.35 per 0.43 mm sample)
    hu_clip: tuple = (0.0, 4000.0)
    far_weight: float = 1.0         # attenuation weight for samples > far_depth_mm behind the trough
    far_depth_mm: float = 50.0      # (far-side jaw, spine: the sources of ghosts); 1.0 = physical
    z_center_vox: float | None = None  # central-ray height at the trough; None = teeth centroid
    mask_threshold: float = 0.5
    batch_frames: int = 12
    channels: dict = field(default_factory=lambda: dict(OPG_CHANNELS))

    @property
    def magnification(self) -> float:
        return self.sid_mm / self.src_trough_mm


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def mu_volume(img: np.ndarray, cfg: OPGConfig) -> np.ndarray:
    """Per-mm attenuation coefficient volume."""
    if cfg.mode == "physical":
        return np.clip(cfg.mu_water_per_mm * (1.0 + img / 1000.0), 0, None).astype(np.float32)
    lo, hi = cfg.hu_clip
    return (np.clip(img, lo, hi) * cfg.simpx_beta * cfg.simpx_ds_per_mm).astype(np.float32)


def channel_bits(lab: np.ndarray, cfg: OPGConfig) -> tuple[list[np.ndarray], list[str]]:
    """Encode channels as bits of float32 volumes, 23 channels per volume (float32 is exact to 2**24).
    One lookup-table pass per volume (labels are uint8). Depth windows are applied per ray at render time."""
    names = list(cfg.channels)
    vols = []
    for c0 in range(0, len(names), 23):
        lut = np.zeros(256, np.int32)
        for b, name in enumerate(names[c0:c0 + 23]):
            for i in cfg.channels[name][0]:
                lut[i] |= 1 << b
        vols.append(lut[lab].astype(np.float32))
    return vols, names


def prepare_labels(lab: np.ndarray, cfg: OPGConfig) -> dict:
    """Per-volume label work, reusable across renders of the same volume with the same channels."""
    bits, names = channel_bits(lab, cfg)
    present = np.bincount(lab.ravel(), minlength=256) > 0
    annotated = {n: bool(present[list(cfg.channels[n][0])].any()) for n in names}
    return {"bits": bits, "names": names, "annotated": annotated,
            "has_teeth": bool(present[TOOTH_IDS].any()),
            "z_teeth": float(np.nonzero(np.isin(lab, TOOTH_IDS))[2].mean()) if present[TOOTH_IDS].any() else None}


def process(image: np.ndarray, px_mm: float, sigma_mm: float = 12.0, amount: float = 0.85,
            gamma: float = 1.0) -> np.ndarray:
    """Vendor-style post-processing: subtract a blurred background, rescale to [0, 1], gamma."""
    out = image - amount * ndimage.gaussian_filter(image, sigma_mm / px_mm)
    lo, hi = np.percentile(out, [0.5, 99.8])
    return np.clip((out - lo) / (hi - lo), 0, 1) ** gamma


def beam_geometry(arch: dict, cfg: OPGConfig) -> dict:
    """Per time step: arch point A (mm), beam direction u, detector axis e, obliquity δ,
    rotation centre C (mm), detector position X (mm). Arrays are in-plane (…, 2)."""
    vx = cfg.voxel_mm
    A = arch["points"] * vx
    n = arch["normals"]
    T = np.gradient(A, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    mid = int(np.argmin(A[:, 1]))                       # most anterior point (j → posterior)
    s = (np.arange(len(A)) - mid) * cfg.column_mm
    if cfg.beam == "normal":
        u = n.copy()
        r = np.full(len(A), np.nan)
    else:
        k_pull = 0.0
        if cfg.beam == "curvature":
            tang = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
            kappa = ndimage.gaussian_filter1d(np.gradient(tang) / cfg.column_mm,
                                              cfg.curvature_smooth_mm / cfg.column_mm)
            r = np.clip(1.0 / np.maximum(np.abs(kappa), 1e-6), cfg.r_min_mm, cfg.r_max_mm)
            k_pull = cfg.k_normal_per_mm
        else:
            a0, a1 = cfg.r_transition_mm
            x = np.clip((np.abs(s) - a0) / (a1 - a0), 0, 1)
            r = cfg.r_front_mm + (cfg.r_back_mm - cfg.r_front_mm) * x * x * (3 - 2 * x)
        phi_n = np.unwrap(np.arctan2(n[:, 1], n[:, 0]))
        phi = np.zeros(len(A))
        phi[mid] = phi_n[mid]
        rot = lambda p: np.array([-np.sin(p), np.cos(p)])
        for f in range(mid + 1, len(A)):
            g = f - 1
            phi[f] = phi[g] + ((T[g] @ rot(phi[g])) / r[g] + k_pull * (phi_n[g] - phi[g])) * cfg.column_mm
        for f in range(mid - 1, -1, -1):
            g = f + 1
            phi[f] = phi[g] - ((T[g] @ rot(phi[g])) / r[g] - k_pull * (phi_n[g] - phi[g])) * cfg.column_mm
        u = np.stack([np.cos(phi), np.sin(phi)], 1)
        dphi = np.gradient(phi) / cfg.column_mm
        r = np.abs((T * np.stack([-u[:, 1], u[:, 0]], 1)).sum(1) / np.where(np.abs(dphi) < 1e-9, 1e-9, dphi))
    e = np.stack([-u[:, 1], u[:, 0]], 1)
    e *= np.sign((e * T).sum(1))[:, None]
    cosd = (e * T).sum(1)
    X = cfg.magnification * np.concatenate([[0], np.cumsum(0.5 * (cosd[1:] + cosd[:-1]) * cfg.column_mm)])
    return {"A": A, "u": u, "e": e, "T": T, "delta_deg": np.degrees(np.arccos(np.clip(cosd, -1, 1))),
            "r": r, "C": A - np.nan_to_num(np.minimum(r, 300))[:, None] * u, "X": X, "s": s, "mid": mid}


def _to_grid(pts_vox: torch.Tensor, shape: tuple) -> torch.Tensor:
    """(…, 3) voxel (i, j, k) → grid_sample coords (x=k, y=j, z=i), align_corners=True."""
    scale = torch.tensor([shape[0] - 1, shape[1] - 1, shape[2] - 1], device=pts_vox.device, dtype=pts_vox.dtype)
    return (2 * pts_vox / scale - 1)[..., [2, 1, 0]]


def render(img: np.ndarray | None, lab: np.ndarray | None, cfg: OPGConfig | None = None,
           arch: dict | None = None, mu: np.ndarray | None = None, labels: dict | None = None) -> dict:
    """Render a volume (LPS). Pass `mu` directly (e.g. a phantom) instead of `img` if wanted.

    Returns image (−log mean T), meanT, masks {channel: bool}, soft masks, geometry, and
    `frame_col`: the image column where the trough point of each time step lands.
    Image rows run superior→inferior; columns run patient right→left.
    """
    cfg = cfg or OPGConfig()
    dev = device()
    vx = cfg.voxel_mm
    if arch is None:
        arch = fit_arch(lab, SimConfig(voxel_mm=vx, column_mm=cfg.column_mm, arch_extend_mm=cfg.arch_extend_mm))
    mu = mu_volume(img, cfg) if mu is None else mu
    shape = mu.shape
    mu_t = torch.from_numpy(mu)[None, None].to(dev)
    names = []
    if lab is not None:
        labels = labels or prepare_labels(lab, cfg)
        names = labels["names"]
        bits_t = [torch.from_numpy(bv)[None, None].to(dev) for bv in labels["bits"]]
    if cfg.z_center_vox is not None:
        zc = cfg.z_center_vox
    elif labels is not None and labels["z_teeth"] is not None:
        zc = labels["z_teeth"]
    else:
        zc = shape[2] / 2

    geo = beam_geometry(arch, cfg)
    M = cfg.magnification
    pd = M * cfg.column_mm                        # detector pixel pitch (mm) = 1 image column
    W = int(round(cfg.slit_mm / pd)) | 1
    offs = (np.arange(W) - W // 2) * pd
    z_top, z_bot = (shape[2] - 1 - zc) * vx + 5, -zc * vx - 5   # object heights at the trough (mm)
    R = int(np.ceil((z_top - z_bot) / cfg.column_mm))
    v = M * (z_top - np.arange(R) * cfg.column_mm)              # detector vertical coords (mm)
    lam = np.arange(cfg.src_trough_mm - cfg.depth_before_mm, cfg.src_trough_mm + cfg.depth_after_mm, cfg.step_mm)
    g = np.deg2rad(cfg.tilt_deg)

    nF = len(geo["A"])
    col0 = geo["X"] / pd + W // 2                 # image column of each frame's slit centre
    ncol = int(np.ceil(col0.max())) + W // 2 + 2
    accT = torch.zeros((R, ncol), device=dev)
    cnt = torch.zeros(ncol, device=dev)
    occ = torch.zeros((len(names), R, ncol), device=dev)

    tf = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    offs_t, v_t, lam_t = tf(offs), tf(v), tf(lam)
    depth = lam - cfg.src_trough_mm                 # signed depth from the focal point (+ = buccal)
    wgt = tf(np.where(depth < -cfg.far_depth_mm, cfg.far_weight, 1.0))
    k0 = int(np.argmin(np.abs(depth)))              # sample at the trough crossing (for the FOV mask)
    hi_vox = tf(np.array(shape) - 1)
    fov = torch.zeros((R, ncol), device=dev)
    # sample indices of each channel's depth window (channels sharing a window are reduced together)
    win_idx = [torch.as_tensor(np.nonzero(np.ones_like(depth, bool) if hw is None else np.abs(depth) <= hw)[0],
                               device=dev) for _, hw in (cfg.channels[n] for n in names)]
    z3 = np.array([0.0, 0.0, 1.0])
    for f0 in range(0, nF, cfg.batch_frames):
        fs = np.arange(f0, min(f0 + cfg.batch_frames, nF))
        u2 = np.concatenate([geo["u"][fs], np.zeros((len(fs), 1))], 1)
        e3 = np.concatenate([geo["e"][fs], np.zeros((len(fs), 1))], 1)
        u = np.cos(g) * u2 + np.sin(g) * z3
        w = -np.sin(g) * u2 + np.cos(g) * z3
        A = np.concatenate([geo["A"][fs], np.full((len(fs), 1), zc * vx)], 1)
        S = A - cfg.src_trough_mm * u
        Dc = S + cfg.sid_mm * u
        S_t, Dc_t, e_t, w_t = tf(S), tf(Dc), tf(e3), tf(w)
        D = (Dc_t[:, None, None] + offs_t[None, :, None, None] * e_t[:, None, None]
             + v_t[None, None, :, None] * w_t[:, None, None])                         # (B, W, R, 3)
        d = D - S_t[:, None, None]
        d = d / d.norm(dim=-1, keepdim=True)
        pts = (S_t[:, None, None, None] + lam_t[None, None, None, :, None] * d[:, :, :, None]) / vx
        grid = _to_grid(pts, shape).reshape(1, -1, 1, 1, 3)
        line = F.grid_sample(mu_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        line = (line.reshape(len(fs), W, R, -1) * wgt).sum(-1) * cfg.step_mm        # (B, W, R)
        p0 = pts[:, :, :, k0]
        inside = ((p0 >= 0) & (p0 <= hi_vox)).all(-1).float().reshape(-1, R).T
        Tr = torch.exp(-line).reshape(-1, R).T                                        # (R, B*W)

        # splat each slit column to its fractional image column (linear weights)
        c = (col0[fs][:, None] + np.arange(W)[None] - W // 2).reshape(-1)
        c_lo = np.floor(c).astype(int)
        w_hi = tf(c - c_lo)
        lo_t, hi_t = torch.as_tensor(c_lo, device=dev), torch.as_tensor(c_lo + 1, device=dev)
        accT.index_add_(1, lo_t, Tr * (1 - w_hi))
        accT.index_add_(1, hi_t, Tr * w_hi)
        cnt.index_add_(0, lo_t, 1 - w_hi)
        cnt.index_add_(0, hi_t, w_hi)
        fov.index_add_(1, lo_t, inside * (1 - w_hi))
        fov.index_add_(1, hi_t, inside * w_hi)
        if names:
            for ci, bt in enumerate(bits_t):
                lb = F.grid_sample(bt, grid, mode="nearest", padding_mode="zeros", align_corners=True)
                lb = lb.reshape(len(fs), W, R, -1).to(torch.int32)
                sliced = {}
                for bb in range(min(23, len(names) - 23 * ci)):
                    b = 23 * ci + bb
                    key = (int(win_idx[b][0]), len(win_idx[b]))
                    if key not in sliced:      # OR of all bits over the window, via max over int samples per bit below
                        sliced[key] = lb.index_select(-1, win_idx[b])
                    hit = ((sliced[key] >> bb) & 1).amax(-1).float().reshape(-1, R).T
                    occ[b].index_add_(1, lo_t, hit * (1 - w_hi))
                    occ[b].index_add_(1, hi_t, hit * w_hi)

    keep = (cnt > 0.5).cpu().numpy()
    meanT = (accT / cnt.clamp(min=1e-6)).cpu().numpy()[:, keep]
    soft = (occ / cnt.clamp(min=1e-6)).cpu().numpy()[:, :, keep]
    fovm = ((fov / cnt.clamp(min=1e-6)).cpu().numpy()[:, keep]) >= 0.5
    image = -np.log(np.clip(meanT, 1e-6, 1))
    first = int(np.argmax(keep))
    frame_col = col0 - first
    ncol_out = int(keep.sum())
    # columns run along increasing arch index; put patient right (low i) on the image left
    flip = geo["A"][0, 0] > geo["A"][-1, 0]
    if flip:
        image, meanT, soft = image[:, ::-1].copy(), meanT[:, ::-1].copy(), soft[:, :, ::-1].copy()
        fovm = fovm[:, ::-1].copy()
        frame_col = ncol_out - 1 - frame_col
    return {
        "image": image, "meanT": meanT, "processed": process(image, cfg.column_mm),
        "masks": {n: soft[b] >= cfg.mask_threshold for b, n in enumerate(names)},
        "soft_masks": {n: soft[b] for b, n in enumerate(names)},
        "fov": fovm,
        "annotated": labels["annotated"] if labels is not None else {},
        "arch": arch, "geometry": geo, "frame_col": frame_col, "z_center_vox": zc, "slit_cols": W,
        "z_top_mm": z_top, "pixel_mm_at_trough": cfg.column_mm,
    }
