#!/usr/bin/env python3
"""Run the active source-seeking simulation used in the paper example."""
from __future__ import annotations

import os

# The simulation uses many small dense linear algebra problems. Limiting BLAS
# threads avoids slow oversubscription on laptops and shared environments.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import math

import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_source_seeking.meshio_utils import load_gmsh_tri
from active_source_seeking.fem_advection_diffusion import assemble_advection_diffusion, time_discretize_implicit_euler
from active_source_seeking.source_model import source_injection_vector
from active_source_seeking.fem_sampling_p1 import fe_p1_triplet
from active_source_seeking.fe_impf import fe_impf_step_free
from active_source_seeking.active_sensing import build_measurement_information_free_sparse
from active_source_seeking.central_controller import bayes_fim_free, doptimal_loss_weighted, grad_loss_one_sensor_free
from active_source_seeking.apf import apf_step, inside_mesh
from active_source_seeking.linalg_utils import cho_factor_spd
from active_source_seeking.pf_utils import uncertainty_radius_2sigma
from active_source_seeking.exact_fim import parse_snapshot_steps, save_exact_fim_snapshot

log = logging.getLogger("active_source_seeking")


# -----------------------
# Logging (console + file)
# -----------------------
def setup_logging(log_path: Path, quiet: bool = False, debug: bool = False):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")

    console_level = logging.WARNING if quiet else (logging.DEBUG if debug else logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(console_level)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG if debug else logging.INFO)

    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.getLogger("skfem").setLevel(logging.WARNING)
    logging.getLogger("skfem.assembly").setLevel(logging.WARNING)
    logging.getLogger("skfem.assembly.basis").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def sanitize_tag(text: str) -> str:
    """Return a filesystem-friendly tag for output folders."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    cleaned = "".join(ch if ch in allowed else "_" for ch in str(text).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "run"


def build_run_tags(preset: str, cfg: "ExperimentConfig", custom_name: str | None = None) -> tuple[str, str]:
    """Build a dated output-folder tag and a shorter plot label."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_tag = (
        f"{preset}"
        f"_T{cfg.N_trials}"
        f"_K{cfg.steps}"
        f"_S{cfg.S}"
        f"_M{cfg.M}"
        f"_D{int(cfg.diffusivity)}"
        f"_rf{cfg.r_filter_val:.4g}"
        f"_{cfg.controller}"
        f"_upd{cfg.motion_update_every}"
        f"_seed{cfg.base_seed}"
    )
    config_tag = sanitize_tag(config_tag)
    if custom_name:
        run_tag = f"{timestamp}_{sanitize_tag(custom_name)}_{config_tag}"
    else:
        run_tag = f"{timestamp}_{config_tag}"
    return run_tag, config_tag


# -----------------------
# Helpers (geometria + FEM sensing)
# -----------------------
def mesh_bbox(mesh):
    xmin, xmax = float(mesh.p[0].min()), float(mesh.p[0].max())
    ymin, ymax = float(mesh.p[1].min()), float(mesh.p[1].max())
    return xmin, xmax, ymin, ymax


def safe_inside_mesh(mesh, finder, p: np.ndarray) -> bool:
    try:
        return bool(inside_mesh(mesh, finder, p))
    except Exception:
        return False


def project_to_mesh(mesh, finder, p_old: np.ndarray, p_new: np.ndarray, n_steps: int = 60) -> np.ndarray:
    """Project a point back into the mesh along the previous motion segment."""
    if safe_inside_mesh(mesh, finder, p_new):
        return p_new
    for a in np.linspace(1.0, 0.0, n_steps):
        p = a * p_new + (1.0 - a) * p_old
        if safe_inside_mesh(mesh, finder, p):
            return p
    return p_old


def sample_point_in_domain(mesh, finder, rng: np.random.Generator):
    xmin, xmax = mesh.p[0].min(), mesh.p[0].max()
    ymin, ymax = mesh.p[1].min(), mesh.p[1].max()
    for _ in range(30000):
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        if safe_inside_mesh(mesh, finder, p):
            return p
    raise RuntimeError("Could not sample interior point.")


def init_sensors_grid_projected(
    mesh,
    finder,
    S: int,
    rng: np.random.Generator,
    nx: int | None = None,
    ny: int | None = None,
    margin: float = 0.02,
    jitter: float = 0.0,
    anchor: np.ndarray | None = None,
):
    """Place sensors on a projected grid inside the mesh."""
    if anchor is None:
        anchor = sample_point_in_domain(mesh, finder, rng)

    if nx is None or ny is None:
        nx = int(math.ceil(math.sqrt(S)))
        ny = int(math.ceil(S / nx))

    xmin, xmax, ymin, ymax = mesh_bbox(mesh)
    dx = xmax - xmin
    dy = ymax - ymin

    xmin2 = xmin + margin * dx
    xmax2 = xmax - margin * dx
    ymin2 = ymin + margin * dy
    ymax2 = ymax - margin * dy

    xs = np.linspace(xmin2, xmax2, nx)
    ys = np.linspace(ymin2, ymax2, ny)

    sensors = np.zeros((S, 2), dtype=float)
    k = 0
    for j in range(ny):
        for i in range(nx):
            if k >= S:
                break
            p = np.array([xs[i], ys[j]], dtype=float)
            if jitter > 0.0:
                p = p + jitter * rng.standard_normal(2)
            sensors[k] = project_to_mesh(mesh, finder, anchor, p)
            k += 1
        if k >= S:
            break

    for i in range(S):
        if not safe_inside_mesh(mesh, finder, sensors[i]):
            sensors[i] = sample_point_in_domain(mesh, finder, rng)
    return sensors


def init_particles_around_rho0(
    mesh,
    finder,
    rng: np.random.Generator,
    M: int,
    rho0: np.ndarray,
    sigma_rho0: float = 200.0,
    u_mu: float = 250.0,
    u_sigma: float = 20.0,
    u_min: float = 1.0,
    u_max: float = 2000.0,
    max_tries: int = 60,
):
    """Initialize particles around a nominal source location and project them inside the mesh."""
    rho0 = np.asarray(rho0, dtype=float).reshape(2)
    z = np.zeros((M, 3), dtype=float)

    for j in range(M):
        p = None
        p_old = rho0.copy()
        for _ in range(max_tries):
            p_try = rho0 + sigma_rho0 * rng.standard_normal(2)
            p_proj = project_to_mesh(mesh, finder, p_old, p_try, n_steps=80)
            if safe_inside_mesh(mesh, finder, p_proj):
                p = p_proj
                break
        if p is None:
            p = sample_point_in_domain(mesh, finder, rng)
        z[j, 0:2] = p

    u = u_mu + u_sigma * rng.standard_normal(M)
    z[:, 2] = np.clip(u, u_min, u_max)

    w = np.full(M, 1.0 / M)
    return z, w


def clip_step(d: np.ndarray, max_step: float) -> np.ndarray:
    n = float(np.linalg.norm(d))
    if n <= max_step:
        return d
    return d * (max_step / (n + 1e-12))


def free_to_full(stepper, x_free: np.ndarray) -> np.ndarray:
    x_full = np.zeros(stepper.N, dtype=float)
    x_full[stepper.free] = x_free
    return x_full


# -----------------------
# -----------------------
def build_g2f(stepper) -> np.ndarray:
    nfree = stepper.free.shape[0]
    g2f = -np.ones(stepper.N, dtype=int)
    g2f[stepper.free] = np.arange(nfree, dtype=int)
    return g2f


class P1PointSampler:
    """
    Precompute FE P1 triplets (tri_nodes, Nhat, free_ids) for a set of fixed points.
    """
    def __init__(self, mesh, finder, stepper, points_xy: np.ndarray):
        self.mesh = mesh
        self.finder = finder
        self.stepper = stepper
        self.points = np.asarray(points_xy, dtype=float)
        self.g2f = build_g2f(stepper)
        P = self.points.shape[0]

        self.tri_nodes = np.zeros((P, 3), dtype=int)
        self.Nhat = np.zeros((P, 3), dtype=float)
        self.free_ids = np.zeros((P, 3), dtype=int)

        for k in range(P):
            tri_nodes, Nhat, _ = fe_p1_triplet(mesh, self.points[k], finder=finder)
            self.tri_nodes[k] = tri_nodes
            self.Nhat[k] = Nhat
            for a in range(3):
                self.free_ids[k, a] = int(self.g2f[int(tri_nodes[a])])

    def eval(self, x_free: np.ndarray) -> np.ndarray:
        # returns values at all points
        P = self.points.shape[0]
        out = np.zeros(P, dtype=float)
        for k in range(P):
            v = 0.0
            for a in range(3):
                ia = int(self.free_ids[k, a])
                xa = float(x_free[ia]) if ia >= 0 else 0.0
                v += float(self.Nhat[k, a]) * xa
            out[k] = v
        return out


def measure_at_point_free(mesh, stepper, x_free: np.ndarray, s_xy: np.ndarray, finder, g2f: np.ndarray) -> float:
    tri_nodes, Nhat, _ = fe_p1_triplet(mesh, s_xy, finder=finder)
    val = 0.0
    for a in range(3):
        g = int(tri_nodes[a])
        ia = int(g2f[g])
        xa = x_free[ia] if ia >= 0 else 0.0
        val += float(Nhat[a]) * float(xa)
    return float(val)


def row_normalize(V, eps=1e-12):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / (n + eps)


# -----------------------
# -----------------------
class GradEMA:
    def __init__(self, S: int, alpha: float = 0.25):
        self.alpha = float(alpha)
        self.g = np.zeros((S, 2), dtype=float)
        self.init = False

    def update(self, g_new: np.ndarray) -> np.ndarray:
        if not self.init:
            self.g[:] = g_new
            self.init = True
        else:
            self.g = (1.0 - self.alpha) * self.g + self.alpha * g_new
        return self.g


def smooth_grad_knn(sensors: np.ndarray, grads: np.ndarray, k: int = 3, lam: float = 0.15) -> np.ndarray:
    S = sensors.shape[0]
    out = grads.copy()
    for i in range(S):
        d = np.linalg.norm(sensors - sensors[i], axis=1)
        nn = np.argsort(d)[1 : k + 1]
        g_nn = grads[nn].mean(axis=0)
        out[i] = (1 - lam) * grads[i] + lam * g_nn
    return out


# -----------------------
# -----------------------
def _make_psd_2x2(H: np.ndarray, min_eig: float = 1e-10) -> np.ndarray:
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, min_eig)
    return (V * w[None, :]) @ V.T


def dogleg_tr_step(g: np.ndarray, H: np.ndarray, Delta: float) -> np.ndarray:
    g = g.reshape(2,)
    H = 0.5 * (H + H.T)

    Hg = H @ g
    gHg = float(g @ Hg)
    gg = float(g @ g)

    if gg < 1e-14:
        return np.zeros(2)

    if gHg <= 1e-12:
        return -Delta * g / (np.sqrt(gg) + 1e-12)

    alpha = gg / (gHg + 1e-12)
    d_u = -alpha * g

    try:
        d_n = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        d_n = d_u

    nu = float(np.linalg.norm(d_u))
    nn = float(np.linalg.norm(d_n))

    if nn <= Delta:
        return d_n
    if nu >= Delta:
        return (Delta / (nu + 1e-12)) * d_u

    p = d_u
    q = d_n - d_u
    a = float(q @ q)
    b = float(2.0 * (p @ q))
    c = float(p @ p - Delta * Delta)
    disc = max(b * b - 4.0 * a * c, 0.0)
    tau = (-b + np.sqrt(disc)) / (2.0 * a + 1e-12)
    tau = float(np.clip(tau, 0.0, 1.0))
    return p + tau * q


def bfgs_update(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    BFGS update for inverse Hessian approximation (we store B ~ Hessian, not inverse):
      B_{k+1} = B_k - (B_k s s^T B_k)/(s^T B_k s) + (y y^T)/(y^T s)
    Here we store B (Hessian approx).
    """
    s = s.reshape(2, 1)
    y = y.reshape(2, 1)

    ys = float((y.T @ s).item())
    if ys <= 1e-10:
        return H  # curvature condition fails -> skip

    Bs = H @ s
    sBs = float((s.T @ Bs).item())
    if sBs <= 1e-12:
        return H

    term1 = (Bs @ Bs.T) / (sBs + 1e-12)
    term2 = (y @ y.T) / (ys + 1e-12)
    Hn = H - term1 + term2
    return _make_psd_2x2(Hn, min_eig=1e-10)


class TrustRegionBFGSController:
    """
    Centralized TR per-sensor with BFGS Hessian approx.
    """
    def __init__(self, S: int, Delta0: float = 80.0, Delta_min: float = 10.0, Delta_max: float = 350.0):
        self.S = int(S)
        self.Delta = float(Delta0)
        self.Delta_min = float(Delta_min)
        self.Delta_max = float(Delta_max)

        self.B = np.stack([np.eye(2) for _ in range(self.S)], axis=0)  # Hessian approx per sensor
        self.g_prev = np.zeros((self.S, 2), dtype=float)
        self.have_prev = False

    def update_radius(self, rho: float):
        if rho < 0.25:
            self.Delta = max(self.Delta_min, 0.5 * self.Delta)
        elif rho > 0.75:
            self.Delta = min(self.Delta_max, 1.5 * self.Delta)

    def propose_step(
        self,
        sensors: np.ndarray,
        grad_all_func,                 # function(sensors_xy)->(S,2) grads
        loss_func,                     # function(sensors_xy)->float
        project_fn,                    # function(p_old,p_new)->p_proj
        hess_damp: float = 5e-2,
        max_inner_tries: int = 2,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Returns sensors_trial, L_trial, rho_ratio, pred_red, act_red.
        """
        L0 = float(loss_func(sensors))
        G0 = np.asarray(grad_all_func(sensors), dtype=float).reshape(self.S, 2)

        # build steps
        d_all = np.zeros_like(sensors)
        pred_red = 0.0

        for i in range(self.S):
            p0 = sensors[i].copy()
            g = G0[i]

            B = _make_psd_2x2(self.B[i], min_eig=1e-10) + hess_damp * np.eye(2)
            d = dogleg_tr_step(g, B, self.Delta)

            pred = -(float(g @ d) + 0.5 * float(d @ (B @ d)))
            pred_red += max(pred, 0.0)

            p1 = project_fn(p0, p0 + d)
            d_all[i] = p1 - p0

        sensors_trial = sensors + d_all
        L1 = float(loss_func(sensors_trial))

        act_red = L0 - L1
        rho = act_red / (pred_red + 1e-12)

        # reject -> shrink and retry
        tries = 0
        while ((rho <= 0.0) or (act_red <= 0.0)) and (tries < max_inner_tries):
            self.Delta = max(self.Delta_min, 0.5 * self.Delta)
            tries += 1

            d_all[:] = 0.0
            pred_red = 0.0
            for i in range(self.S):
                p0 = sensors[i].copy()
                g = G0[i]
                B = _make_psd_2x2(self.B[i], min_eig=1e-10) + hess_damp * np.eye(2)
                d = dogleg_tr_step(g, B, self.Delta)
                pred = -(float(g @ d) + 0.5 * float(d @ (B @ d)))
                pred_red += max(pred, 0.0)
                p1 = project_fn(p0, p0 + d)
                d_all[i] = p1 - p0

            sensors_trial = sensors + d_all
            L1 = float(loss_func(sensors_trial))
            act_red = L0 - L1
            rho = act_red / (pred_red + 1e-12)

        # update radius anyway
        self.update_radius(rho)

        if (rho > 0.0) and (act_red > 0.0):
            G1 = np.asarray(grad_all_func(sensors_trial), dtype=float).reshape(self.S, 2)
            for i in range(self.S):
                s = (sensors_trial[i] - sensors[i]).reshape(2,)
                if float(np.linalg.norm(s)) < 1e-9:
                    continue
                y = (G1[i] - G0[i]).reshape(2,)
                self.B[i] = bfgs_update(self.B[i], s, y)
            self.g_prev[:] = G1
            self.have_prev = True
        else:
            self.g_prev[:] = G0
            self.have_prev = True

        return sensors_trial, L1, rho, pred_red, act_red


# -----------------------
# -----------------------
def spread_loss_and_grad(sensors: np.ndarray, d_ref: float = 600.0, eps: float = 1e-6) -> Tuple[float, np.ndarray]:
    """
    Repulsive smooth penalty to avoid clustering:
      L = sum_{i<j} exp(-(d_ij/d_ref)^2)
    Grad wrt sensor i:
      dL/dp_i = sum_{j!=i} exp(-(d^2/d_ref^2)) * (-2/d_ref^2) * (p_i - p_j)
    """
    S = sensors.shape[0]
    grad = np.zeros((S, 2), dtype=float)
    loss = 0.0
    inv = 1.0 / (d_ref * d_ref + eps)

    for i in range(S):
        for j in range(i + 1, S):
            dp = sensors[i] - sensors[j]
            d2 = float(dp @ dp)
            w = math.exp(-d2 * inv)
            loss += w
            g = w * (-2.0 * inv) * dp
            grad[i] += g
            grad[j] -= g
    return float(loss), grad


# -----------------------
# Animation helper
# -----------------------
def save_animation(
    mesh,
    field_hist_full: np.ndarray,
    sensors_hist: np.ndarray,
    rho_true: np.ndarray,
    rho_map_hist: np.ndarray,
    map_errors: np.ndarray,
    rho_mmse_hist: np.ndarray,
    mmse_errors: np.ndarray,
    out_path: Path,
    fps: int = 6,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tri = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    fig, ax = plt.subplots()

    tpc = ax.tripcolor(tri, field_hist_full[0], shading="gouraud")
    cb = fig.colorbar(tpc, ax=ax)
    cb.set_label("concentration")
    tpc.set_zorder(0)

    sc = ax.scatter(
        sensors_hist[0][:, 0], sensors_hist[0][:, 1],
        s=40, marker="o", edgecolors="k", linewidths=0.5,
        c="magenta", zorder=5
    )

    h_true = ax.scatter(
        [rho_true[0]], [rho_true[1]],
        marker="*", s=120, c="orange",
        edgecolors="k", linewidths=0.5,
        zorder=30
    )

    trail = ax.scatter(
        rho_map_hist[:1, 0], rho_map_hist[:1, 1],
        s=30, marker="x", c="0.6", linewidths=1.0,
        zorder=10
    )

    h_map = ax.scatter(
        [rho_map_hist[0, 0]], [rho_map_hist[0, 1]],
        s=90, c="red", marker="x", linewidths=2.0,
        zorder=40
    )

    mmse_trail = ax.scatter(
        rho_mmse_hist[:1, 0], rho_mmse_hist[:1, 1],
        s=30, marker="+", c="black", linewidths=1.0,
        zorder=10
    )

    h_mmse = ax.scatter(
        [rho_mmse_hist[0, 0]], [rho_mmse_hist[0, 1]],
        s=80, c="limegreen", marker="+", linewidths=2.0,
        zorder=35
    )

    ax.set_aspect("equal")
    ax.set_xlim(mesh.p[0].min(), mesh.p[0].max())
    ax.set_ylim(mesh.p[1].min(), mesh.p[1].max())

    T = field_hist_full.shape[0]

    def update(k):
        tpc.set_array(field_hist_full[k])
        sc.set_offsets(sensors_hist[k])

        trail.set_offsets(rho_map_hist[: k + 1, :])
        h_map.set_offsets(rho_map_hist[k : k + 1, :])

        mmse_trail.set_offsets(rho_mmse_hist[: k + 1, :])
        h_mmse.set_offsets(rho_mmse_hist[k : k + 1, :])

        ax.set_title(
            f"k={k+1:03d}/{T:03d} | MAP err={float(map_errors[k]):.1f} m | MMSE err={float(mmse_errors[k]):.1f} m"
        )
        return (tpc, sc, trail, h_map, mmse_trail, h_mmse, h_true)

    ani = FuncAnimation(fig, update, frames=T, interval=int(1000 / fps), blit=False)

    try:
        from matplotlib.animation import FFMpegWriter
        ani.save(out_path.with_suffix(".mp4"), writer=FFMpegWriter(fps=fps))
        log.info("Saved animation: %s", out_path.with_suffix(".mp4"))
    except Exception as e:
        log.warning("MP4 save failed (%s). Falling back to GIF.", str(e))
        try:
            ani.save(out_path.with_suffix(".gif"), writer="pillow", fps=fps)
            log.info("Saved animation: %s", out_path.with_suffix(".gif"))
        except Exception as e2:
            log.error("GIF save failed (%s). Animation not saved.", str(e2))

    plt.close(fig)


# -----------------------
# Config + metrics aggregation
# -----------------------
@dataclass
class ExperimentConfig:
    # Monte Carlo
    N_trials: int = 50
    steps: int = 400
    base_seed: int = 123

    # Mesh/PDE
    mesh_path: str = "mesh/bay_port_island.msh"
    dirichlet_tag: str = "open_sea_boundary"
    dt: float = 5.0
    diffusivity: float = 1500.0
    sigma_w_true: float = 1e-4
    sigma_w_filter: float = 2e-2

    # Sensors
    S: int = 10
    r_true_val: float = 1e-5
    r_filter_val: float = 1e-4

    rho_true_mode: str = "random"          # "random" | "given"
    rho_true_given: Tuple[float, float] = (1500.0, 1500.0)
    u_true: float = 250.0

    # PF
    M: int = 250
    u_min: float = 1.0
    u_max: float = 500.0
    sigma_rho_init: float = 25.0
    sigma_logu: float = 0.10
    resample_frac: float = 0.4

    # PF init around rho0
    use_rho0_init: bool = False
    rho0_mode: str = "random"      # "center" | "random" | "given"
    rho0_given: Tuple[float, float] = (4500.0, 1000.0)
    sigma_rho0: float = 1500.0
    u0_mu: float = 250.0
    u0_sigma: float = 60.0
    w_u: float = 1.0

    # Controller objective
    delta: float = 1e-4

    # APF motion
    dt_move: float = 8.0
    alpha_attr: float = 1.6
    k_rep_agents: float = 2e4
    k_rep_bbox: float = 2e4
    d0_agents: float = 450.0
    d0_bbox: float = 200.0
    max_speed: float = 450.0
    max_step_per_iter: float = 350.0

    # TR-BFGS
    tr_Delta0: float = 150.0
    tr_Delta_min: float = 20.0
    tr_Delta_max: float = 400.0
    tr_hess_damp: float = 1e-2
    tr_max_inner_tries: int = 2

    # Motion regularization (slew / smoothness)
    beta_seek_max: float = 1e-6
    beta_seek_scale: float = 2000.0
    beta_slew: float = 0.0

    spread_beta: float = 0.0      # optional spacing regularizer
    spread_dref: float = 600.0    # [m] reference distance for exp penalty

    grad_ema_alpha: float = 0.20
    grad_knn_k: int = 3
    grad_knn_lam: float = 0.15

    # Field error sampling
    P_field: int = 200
    field_error_every: int = 1

    # Field error metric. Use "relative" for the normalized RMSE reported in the paper.
    field_error_metric: str = "relative"

    # Only used by "weighted"
    field_wgt_eps: float = 1e-12

    # Output and logging
    save_trial_trajs: int = 1
    anim_fps: int = 6
    log_every: int = 25
    debug_logs: bool = False

    # Speed/teaching options. The paper configuration uses trbfgs with
    # motion_update_every=1. Larger values reuse the last informative
    # direction for a few steps and can greatly reduce runtime.
    controller: str = "trbfgs"       # "trbfgs" or "gradient"
    motion_update_every: int = 1

    # Exact posterior-FIM snapshot maps. Disabled by default because each map
    # evaluates Fbar=sum_j w_j G_j.T @ Psi @ G_j over a candidate-location grid.
    save_exact_fim: bool = False
    exact_fim_trials: int = 1
    exact_fim_steps: str = "1,30,60,110,180,240,last"
    exact_fim_nx: int = 95
    exact_fim_ny: int = 70


def compute_quantiles(x: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95), axis=0) -> Dict[str, np.ndarray]:
    out = {}
    qv = np.quantile(x, qs, axis=axis)
    for i, q in enumerate(qs):
        out[f"q{int(round(q * 100)):02d}"] = qv[i]
    return out


def rmse_over_trials(err: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean(err ** 2, axis=axis))


# -----------------------
# One trial run
# -----------------------
def run_one_trial(
    mesh,
    boundaries,
    finder,
    stepper_true,
    stepper_filt,
    cfg: ExperimentConfig,
    seed: int,
    field_points: np.ndarray,
    field_sampler_true: P1PointSampler,
    field_sampler_filt: P1PointSampler,
    trial_index: int,
    trial_dir: Path,
) -> Dict[str, Any]:

    rng = np.random.default_rng(seed)
    nfree = stepper_true.free.shape[0]
    S = cfg.S
    M = cfg.M
    steps = cfg.steps

    g2f_true = build_g2f(stepper_true)
    g2f_filt = build_g2f(stepper_filt)

    # -----------------------
    # -----------------------
    if cfg.rho_true_mode == "random":
        rho_true = sample_point_in_domain(mesh, finder, rng)
    elif cfg.rho_true_mode == "given":
        rho_true = np.array(cfg.rho_true_given, dtype=float)
    else:
        raise ValueError(f"Unknown rho_true_mode={cfg.rho_true_mode}")
    rho_true = project_to_mesh(mesh, finder, rho_true, rho_true, n_steps=120)

    u_true = float(cfg.u_true)
    x_true_free = np.zeros(nfree, dtype=float)

    # --- sensors init (grid projected) ---
    anchor = sample_point_in_domain(mesh, finder, rng)
    sensors = init_sensors_grid_projected(
        mesh=mesh, finder=finder, S=S, rng=rng,
        nx=None, ny=None, margin=0.02, jitter=0.0, anchor=anchor
    )

    r_true = cfg.r_true_val * np.ones(S)
    r_filter = cfg.r_filter_val * np.ones(S)

    # --- PF init ---
    u_min, u_max = cfg.u_min, cfg.u_max
    if cfg.use_rho0_init:
        if cfg.rho0_mode == "center":
            rho0 = np.array(
                [0.5 * (mesh.p[0].min() + mesh.p[0].max()),
                 0.5 * (mesh.p[1].min() + mesh.p[1].max())],
                dtype=float
            )
        elif cfg.rho0_mode == "random":
            rho0 = sample_point_in_domain(mesh, finder, rng)
        elif cfg.rho0_mode == "given":
            rho0 = np.array(cfg.rho0_given, dtype=float)
        else:
            raise ValueError(f"Unknown rho0_mode={cfg.rho0_mode}")

        rho0 = project_to_mesh(mesh, finder, rho0, rho0, n_steps=120)
        z_prev, w_prev = init_particles_around_rho0(
            mesh=mesh, finder=finder, rng=rng, M=M, rho0=rho0,
            sigma_rho0=cfg.sigma_rho0,
            u_mu=cfg.u0_mu, u_sigma=cfg.u0_sigma,
            u_min=cfg.u_min, u_max=cfg.u_max,
        )
    else:
        z_prev = np.zeros((M, 3), dtype=float)
        for j in range(M):
            z_prev[j, 0:2] = sample_point_in_domain(mesh, finder, rng)
            z_prev[j, 2] = rng.uniform(cfg.u_min, cfg.u_max)
        w_prev = np.full(M, 1.0 / M)

    Omega_prev = np.diag(stepper_filt.Qinv_free.diagonal().astype(float))
    q_prev = np.zeros((M, nfree), dtype=float)
    bbar_free = np.zeros(nfree, dtype=float)

    sigma_rho = float(cfg.sigma_rho_init)
    sigma_logu = float(cfg.sigma_logu)
    resample_threshold = cfg.resample_frac * M

    # --- controller state ---
    tr = TrustRegionBFGSController(
        S=S, Delta0=cfg.tr_Delta0, Delta_min=cfg.tr_Delta_min, Delta_max=cfg.tr_Delta_max
    )
    grad_ema = GradEMA(S, alpha=cfg.grad_ema_alpha)

    # --- histories per step ---
    map_pos_err = np.zeros(steps)
    mmse_pos_err = np.zeros(steps)
    map_u_err = np.zeros(steps)
    mmse_u_err = np.zeros(steps)

    map_field_mse = np.full(steps, np.nan)
    mmse_field_mse = np.full(steps, np.nan)

    # --- optional saving (first K trials) ---
    save_traj = (trial_index < cfg.save_trial_trajs)
    traj = {}
    if save_traj:
        traj["sensors"] = np.zeros((steps, S, 2))
        traj["rho_map"] = np.zeros((steps, 2))
        traj["rho_mmse"] = np.zeros((steps, 2))
        traj["u_map"] = np.zeros((steps,))
        traj["u_mmse"] = np.zeros((steps,))
        traj["rho_true"] = rho_true.copy()
        traj["u_true"] = u_true

        field_hist_full = np.zeros((steps, stepper_true.N), dtype=float)
        sensors_hist = np.zeros((steps, S, 2), dtype=float)
        rho_map_hist = np.zeros((steps, 2), dtype=float)
        rho_mmse_hist = np.zeros((steps, 2), dtype=float)
        map_errors = np.zeros((steps,), dtype=float)
        mmse_errors = np.zeros((steps,), dtype=float)

    t_start = time.time()
    last_dirs_info = np.zeros((S, 2), dtype=float)
    have_last_dirs = False
    exact_fim_steps = parse_snapshot_steps(cfg.exact_fim_steps, steps)
    save_exact_fim_this_trial = bool(cfg.save_exact_fim) and (trial_index < int(cfg.exact_fim_trials))

    def make_loss_func(
        mesh, stepper_filt, finder,
        z, w, Omega_k,
        y, r_filter,
        delta,
        rho_target: np.ndarray | None = None,
        beta_seek: float = 0.0,
        s_ref: np.ndarray | None = None,
        beta_slew: float = 0.0,
    ):
        def loss_func(sensors_xy: np.ndarray) -> float:
            Lambda_sp, lam = build_measurement_information_free_sparse(
                mesh, stepper_filt, sensors_xy, y, r_filter, finder=finder
            )
            Lambda_sp2 = 0.5 * (Lambda_sp + Lambda_sp.T)

            Omega = 0.5 * (Omega_k + Omega_k.T)
            cF, Omega_spd, _ = cho_factor_spd(Omega, base_jitter=1e-14, max_tries=12)
            Lambda_dense = Lambda_sp2.toarray()
            X = la.cho_solve(cF, Lambda_dense, check_finite=False)
            Psi_dense = Lambda_dense - Lambda_dense @ X

            Fbar = bayes_fim_free(mesh, stepper_filt, Psi_dense, z, w, finder=finder)
            L_total = float(doptimal_loss_weighted(Fbar, delta, w_u=cfg.w_u))

            if (rho_target is not None) and (beta_seek > 0.0):
                ds_seek = sensors_xy - np.asarray(rho_target, dtype=float).reshape(1, 2)
                L_total += float(beta_seek * np.sum(ds_seek * ds_seek))

            if (s_ref is not None) and (beta_slew > 0.0):
                ds_slew = sensors_xy - s_ref
                L_total += float(beta_slew * np.sum(ds_slew * ds_slew))

            return L_total

        return loss_func

    def grad_all_func_factory(z, w, Omega_k, y):
        """
        Returns grad_all(sensors)->(S,2) that includes:
         - D-opt gradient via grad_loss_one_sensor_free
         - + spread gradient (analytic)
        """
        def grad_all(sensors_xy: np.ndarray) -> np.ndarray:
            Lambda_sp_g, lam_g = build_measurement_information_free_sparse(
                mesh, stepper_filt, sensors_xy, y, r_filter, finder=finder
            )
            Lambda_sp_g = 0.5 * (Lambda_sp_g + Lambda_sp_g.T)

            Omega = 0.5 * (Omega_k + Omega_k.T)
            cF, Omega_spd, _ = cho_factor_spd(Omega, base_jitter=1e-14, max_tries=12)

            Lambda_dense = Lambda_sp_g.toarray()
            X = la.cho_solve(cF, Lambda_dense, check_finite=False)
            Psi_dense = Lambda_dense - Lambda_dense @ X

            G = np.zeros((S, 2), dtype=float)
            for i in range(S):
                gi = grad_loss_one_sensor_free(
                    mesh=mesh, stepper=stepper_filt, sensors_xy=sensors_xy,
                    r=r_filter, z=z, w=w, Omega=Omega_spd,
                    Lambda_sp=Lambda_sp_g, lam=lam_g, delta=cfg.delta,
                    i=i, finder=finder, cF=cF, X=X, Psi=Psi_dense,
                    objective="dopt", w_u=cfg.w_u,
                )
                G[i] = np.asarray(gi, dtype=float).reshape(2,)

            # add spread regularizer gradient
            _, Gs = spread_loss_and_grad(sensors_xy, d_ref=cfg.spread_dref)
            G = G + float(cfg.spread_beta) * Gs
            return G

        return grad_all

    for k in range(steps):
        # ---- simulate true field ----
        b_full, _, _ = source_injection_vector(mesh, rho_true, finder=finder)
        b_free = stepper_true.restrict_full(b_full)
        x_true_free = (stepper_true.A_free @ x_true_free) + stepper_true.solve_free(cfg.dt * b_free * u_true)

        if save_traj:
            field_hist_full[k] = free_to_full(stepper_true, x_true_free)

        # ---- measurements ----
        y = np.zeros(S, dtype=float)
        for i in range(S):
            if not safe_inside_mesh(mesh, finder, sensors[i]):
                sensors[i] = sample_point_in_domain(mesh, finder, rng)
            yi = measure_at_point_free(mesh, stepper_true, x_true_free, sensors[i], finder=finder, g2f=g2f_true)
            y[i] = yi + np.sqrt(r_true[i]) * rng.standard_normal()

        # ---- centralized info ----
        Lambda_sp, lam = build_measurement_information_free_sparse(
            mesh, stepper_filt, sensors, y, r_filter, finder=finder
        )
        Lambda_sp = 0.5 * (Lambda_sp + Lambda_sp.T)

        # ---- FE-IMPF update ----
        z, w, q_prev, Omega_k, Omega_pred, x_pred, Psi = fe_impf_step_free(
            mesh=mesh,
            stepper=stepper_filt,
            dt=cfg.dt,
            Omega_prev=Omega_prev,
            q_prev=q_prev,
            z_prev=z_prev,
            w_prev=w_prev,
            Lambda_sp=Lambda_sp,
            lam=lam,
            bbar_free=bbar_free,
            rng=rng,
            finder=finder,
            resample_threshold=resample_threshold,
            sigma_rho=sigma_rho,
            sigma_logu=sigma_logu,
            u_min=u_min,
            u_max=u_max,
            u0_mu=cfg.u0_mu,
        )
        Omega_prev = Omega_k
        z_prev, w_prev = z, w

        w = np.asarray(w).ravel()
        w = w / (w.sum() + 1e-12)

        # ---- MAP/MMSE ----
        jbest = int(np.argmax(w))
        rho_map = z[jbest, 0:2].copy()
        u_map = float(z[jbest, 2])

        rho_mmse = (w[:, None] * z[:, 0:2]).sum(axis=0)
        u_mmse = float((w * z[:, 2]).sum())

        map_pos_err[k] = float(np.linalg.norm(rho_map - rho_true))
        mmse_pos_err[k] = float(np.linalg.norm(rho_mmse - rho_true))
        map_u_err[k] = abs(u_map - u_true)
        mmse_u_err[k] = abs(u_mmse - u_true)

        if save_exact_fim_this_trial and (k in exact_fim_steps):
            fim_dir = trial_dir / f"trial_{trial_index:03d}_exact_fim"
            fim_path = save_exact_fim_snapshot(
                out_dir=fim_dir,
                k=k,
                mesh=mesh,
                stepper=stepper_filt,
                sensors_xy=sensors,
                y=y,
                r_filter=r_filter,
                z=z,
                w=w,
                Omega_k=Omega_k,
                finder=finder,
                delta=cfg.delta,
                w_u=cfg.w_u,
                rho_map=rho_map,
                rho_mmse=rho_mmse,
                rho_true=rho_true,
                nx=cfg.exact_fim_nx,
                ny=cfg.exact_fim_ny,
            )
            log.info("Saved exact posterior-FIM snapshot: %s", fim_path)

        # --- particle-filter diagnostics ---
        ess = 1.0 / (np.sum(w**2) + 1e-12)
        logu = np.log(np.clip(z[:, 2], u_min, u_max))
        var_logu = float(np.sum(w * (logu - np.sum(w * logu))**2))
        d = np.linalg.norm(z[:, 0:2] - rho_mmse[None, :], axis=1)  # dist-to-MMSE-rho per particle

        # weighted corr(u, dist-to-MMSE-rho)
        u = z[:, 2]
        mu_u = float(np.sum(w * u)); mu_d = float(np.sum(w * d))
        cov_ud = float(np.sum(w * (u - mu_u) * (d - mu_d)))
        var_u  = float(np.sum(w * (u - mu_u)**2)); var_d = float(np.sum(w * (d - mu_d)**2))
        corr_ud = cov_ud / (np.sqrt(var_u * var_d) + 1e-12)

        log.debug("PF diag | ESS=%.1f/%d | var(logu)=%.3e", ess, M, var_logu)
        log.debug("PF diag | corr(u, dist_to_rhoMMSE)=%.3f", corr_ud)

        # ---- field error on sampled points ----
        if (k % cfg.field_error_every) == 0:
            x_map_free = x_pred[jbest].copy()
            x_mmse_free = (w[:, None] * x_pred).sum(axis=0)

            yt = field_sampler_true.eval(x_true_free)      # ground-truth at sampled points
            ym = field_sampler_filt.eval(x_map_free)       # MAP-particle predicted field
            yq = field_sampler_filt.eval(x_mmse_free)      # MMSE predicted field

            metric = str(getattr(cfg, "field_error_metric", "plain")).lower()

            if metric == "weighted":
                # weights emphasize regions where concentration is larger
                den0 = float(np.mean(yt**2)) + float(getattr(cfg, "field_wgt_eps", 1e-12))
                wgt = (yt**2) / den0  # normalized weights
                wsum = float(np.sum(wgt)) + 1e-12

                map_field_mse[k]  = float(np.sum(wgt * (ym - yt)**2) / wsum)
                mmse_field_mse[k] = float(np.sum(wgt * (yq - yt)**2) / wsum)

            elif metric == "relative":
                den = float(np.mean(yt**2)) + 1e-12

                map_field_mse[k]  = float(np.mean((ym - yt)**2) / den)
                mmse_field_mse[k] = float(np.mean((yq - yt)**2) / den)

            else:
                # "plain"
                map_field_mse[k]  = float(np.mean((ym - yt) ** 2))
                mmse_field_mse[k] = float(np.mean((yq - yt) ** 2))

        # ============================================================
        # Motion update
        # ============================================================
        s_ref = sensors.copy()
        rad_2sig = float(uncertainty_radius_2sigma(z, w))
        beta_seek = float(cfg.beta_seek_max / (1.0 + (rad_2sig / cfg.beta_seek_scale) ** 2))

        # Build the dual objective: information loss plus source-seeking term.
        loss_func = make_loss_func(
            mesh=mesh,
            stepper_filt=stepper_filt,
            finder=finder,
            z=z, w=w, Omega_k=Omega_k,
            y=y, r_filter=r_filter,
            delta=cfg.delta,
            rho_target=rho_mmse,
            beta_seek=beta_seek,
            s_ref=s_ref,
            beta_slew=cfg.beta_slew,
        )

        # Base gradient (info + spread) factory
        grad_all_raw = grad_all_func_factory(z=z, w=w, Omega_k=Omega_k, y=y)

        def grad_all_with_dual_terms(sensors_xy: np.ndarray) -> np.ndarray:
            G = grad_all_raw(sensors_xy)
            if beta_seek > 0.0:
                G = G + (2.0 * beta_seek) * (sensors_xy - rho_mmse[None, :])
            if cfg.beta_slew > 0.0:
                G = G + (2.0 * cfg.beta_slew) * (sensors_xy - s_ref)
            return G

        # ============================================================
        #           (do NOT smooth grads at trial points -> avoids biasing rho-test)
        # ============================================================
        def grad_all_smoothed(sensors_xy: np.ndarray) -> np.ndarray:
            G = grad_all_with_dual_terms(sensors_xy)
            # smooth only at the current iterate (exactly), not at trial points
            if np.allclose(sensors_xy, sensors):
                G = grad_ema.update(G)
                G = smooth_grad_knn(sensors_xy, G, k=cfg.grad_knn_k, lam=cfg.grad_knn_lam)
            return G

        def proj_fn(p_old, p_new):
            return project_to_mesh(mesh, finder, p_old, p_new)

        do_motion_update = (not have_last_dirs) or ((k % max(1, int(cfg.motion_update_every))) == 0)
        L_trial = np.nan
        rho_ratio = np.nan
        pred_red = np.nan
        act_red = np.nan

        if do_motion_update:
            controller_name = str(cfg.controller).lower()
            if controller_name == "gradient":
                # Fast first-order direction: one gradient evaluation, no trust-region
                # acceptance test and no BFGS curvature update. Useful for teaching runs.
                G0 = np.asarray(grad_all_smoothed(sensors), dtype=float).reshape(S, 2)
                d_info = -G0
                dirs_info = row_normalize(d_info)
            else:
                # Paper-style TR-BFGS update. This is accurate but expensive because it
                # evaluates objective/gradients at trial points.
                sensors_trial, L_trial, rho_ratio, pred_red, act_red = tr.propose_step(
                    sensors=sensors,
                    grad_all_func=grad_all_smoothed,
                    loss_func=loss_func,
                    project_fn=proj_fn,
                    hess_damp=cfg.tr_hess_damp,
                    max_inner_tries=cfg.tr_max_inner_tries,
                )
                d_info = sensors_trial - sensors
                dirs_info = row_normalize(d_info)

            last_dirs_info[:] = dirs_info
            have_last_dirs = True
        else:
            # Reuse last informative direction; the source-seeking component below is
            # still recomputed from the current MMSE estimate at every step.
            dirs_info = last_dirs_info.copy()

        # Normalize the active-sensing direction and gradually increase attraction
        # toward the MMSE estimate as the posterior uncertainty contracts.
        gamma = float(np.clip(1.0 - rad_2sig / cfg.beta_seek_scale, 0.12, 0.55))

        to_target = rho_mmse[None, :] - sensors
        dirs_seek = row_normalize(to_target)

        dirs = row_normalize((1.0 - gamma) * dirs_info + gamma * dirs_seek)

        gain = float(np.clip(rad_2sig / 1400.0, 0.35, 1.10))
        dirs = row_normalize(gain * dirs)


        # ---- APF motion ----
        sensors_prev = sensors.copy()
        sensors = apf_step(
            mesh=mesh,
            finder=finder,
            sensors=sensors,
            grad_dirs=dirs,
            dt_move=cfg.dt_move * gain,
            alpha_attr=cfg.alpha_attr * gain,
            d0_agents=cfg.d0_agents,
            k_rep_agents=cfg.k_rep_agents,
            d0_bbox=cfg.d0_bbox,
            k_rep_bbox=cfg.k_rep_bbox,
            max_speed=cfg.max_speed * gain,
            rng=rng,
        )

        slew = 0.70
        sensors = (1.0 - slew) * sensors_prev + slew * sensors

        # repair + cap + project robustly
        sensors_before_cap = sensors.copy()
        step_raw = np.linalg.norm(sensors_before_cap - sensors_prev, axis=1)

        for i in range(S):
            if not safe_inside_mesh(mesh, finder, sensors[i]):
                sensors[i] = sample_point_in_domain(mesh, finder, rng)

        for i in range(S):
            step_vec = sensors[i] - sensors_prev[i]
            step_vec = clip_step(step_vec, max_step=cfg.max_step_per_iter)
            sensors[i] = project_to_mesh(mesh, finder, sensors_prev[i], sensors_prev[i] + step_vec)

        step_cap = np.linalg.norm(sensors - sensors_prev, axis=1)
        log.debug(
            "Sensor step | raw mean/max=%.1f/%.1f m | capped/projected mean/max=%.1f/%.1f m",
            float(step_raw.mean()), float(step_raw.max()),
            float(step_cap.mean()), float(step_cap.max()),
        )

        # ---- store trajectories ----
        if save_traj:
            traj["sensors"][k] = sensors
            traj["rho_map"][k] = rho_map
            traj["rho_mmse"][k] = rho_mmse
            traj["u_map"][k] = u_map
            traj["u_mmse"][k] = u_mmse

            sensors_hist[k] = sensors
            rho_map_hist[k] = rho_map
            rho_mmse_hist[k] = rho_mmse
            map_errors[k] = map_pos_err[k]
            mmse_errors[k] = mmse_pos_err[k]

        # ---- logs ----
        step = k + 1
        if (step == 1) or (cfg.log_every > 0 and step % cfg.log_every == 0) or (step == steps):
            elapsed = time.time() - t_start
            if str(cfg.controller).lower() == "trbfgs" and np.isfinite(rho_ratio):
                log.info(
                    "TR-BFGS: Delta=%.1f rho=%.2f pred=%.2e act=%.2e | L=%.2e",
                    tr.Delta, rho_ratio, pred_red, act_red, float(L_trial)
                )
            elif str(cfg.controller).lower() == "trbfgs":
                log.info("TR-BFGS: skipped active-sensing update this step; reused last direction")
            else:
                msg = "gradient controller: recomputed direction" if do_motion_update else "gradient controller: reused last direction"
                log.info(msg)
            log.info(
                "Trial %d | k=%03d/%03d | MAPerr=%.1f m | MMSEerr=%.1f m | du(MAP)=%.2f | t=%.1fs",
                trial_index + 1, step, steps, map_pos_err[k], mmse_pos_err[k], map_u_err[k], elapsed
            )

    if (k == 0) or (k % 50 == 0):
        yabs = np.abs(y)
        log.info("y stats | min=%.3e med=%.3e max=%.3e", float(yabs.min()), float(np.median(yabs)), float(yabs.max()))

    # ---- Save trial package for animation ----
    if save_traj:
        trial_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trial_dir / f"trial_{trial_index:03d}.npz",
            field_hist_full=field_hist_full,
            sensors_hist=sensors_hist,
            rho_true=rho_true,
            rho_map_hist=rho_map_hist,
            rho_mmse_hist=rho_mmse_hist,
            map_errors=map_errors,
            mmse_errors=mmse_errors,
            u_true=float(u_true),
            dt=float(cfg.dt),
            diffusivity=float(cfg.diffusivity),
            S=int(cfg.S),
            M=int(cfg.M),
            r_filter_val=float(cfg.r_filter_val),
            seed=int(seed),
        )
        log.info("Saved trial history for animation: %s", trial_dir / f"trial_{trial_index:03d}.npz")

    out = {
        "rho_true": rho_true,
        "u_true": u_true,
        "map_pos_err": map_pos_err,
        "mmse_pos_err": mmse_pos_err,
        "map_u_err": map_u_err,
        "mmse_u_err": mmse_u_err,
        "map_field_mse": map_field_mse,
        "mmse_field_mse": mmse_field_mse,
    }
    if save_traj:
        out["traj"] = traj
    return out


# -----------------------
# Plot utilities
# -----------------------
def plot_with_bands(
    x_mean,
    qdict: Dict[str, np.ndarray],
    title: str,
    ylabel: str,
    out_png: Path,
    n_trials: int | None = None,
):
    """Plot Monte Carlo mean and quantile bands.

    With a single trial, quantile bands are not informative, so only the
    realized error curve is shown.
    """
    plt.figure()
    k = np.arange(len(x_mean)) + 1
    curve_label = "single trial" if n_trials == 1 else "mean"
    plt.plot(k, x_mean, label=curve_label)
    if n_trials is None or n_trials > 1:
        if "q25" in qdict and "q75" in qdict:
            plt.fill_between(k, qdict["q25"], qdict["q75"], alpha=0.25, label="25-75%")
        if "q05" in qdict and "q95" in qdict:
            plt.fill_between(k, qdict["q05"], qdict["q95"], alpha=0.15, label="5-95%")
    plt.title(title)
    plt.xlabel("step k")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def generate_animations_for_saved_trials(mesh, trial_dir: Path, out_dir: Path, fps: int):
    if not trial_dir.exists():
        return
    files = sorted(trial_dir.glob("trial_*.npz"))
    for f in files:
        data = np.load(f, allow_pickle=True)
        out_path = out_dir / f"anim_{f.stem}"
        log.info("Generating animation from: %s", f)
        save_animation(
            mesh=mesh,
            field_hist_full=data["field_hist_full"],
            sensors_hist=data["sensors_hist"],
            rho_true=data["rho_true"],
            rho_map_hist=data["rho_map_hist"],
            map_errors=data["map_errors"],
            rho_mmse_hist=data["rho_mmse_hist"],
            mmse_errors=data["mmse_errors"],
            out_path=out_path,
            fps=fps,
        )


# -----------------------
# Main: configs sweep + MC aggregate
# -----------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Active source-seeking simulation in a 2-D spatio-temporal field."
    )
    parser.add_argument(
        "--preset",
        choices=("demo", "paper", "reference"),
        default="demo",
        help="demo is fast; paper uses the full Monte Carlo setup; reference is a deterministic single-run setup.",
    )
    parser.add_argument("--trials", type=int, default=None, help="Number of Monte Carlo trials.")
    parser.add_argument("--steps", type=int, default=None, help="Number of time steps per trial.")
    parser.add_argument("--particles", type=int, default=None, help="Number of source particles.")
    parser.add_argument("--sensors", type=int, default=None, help="Number of mobile sensors.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
    parser.add_argument("--output", type=str, default=None, help="Base output directory. Default: outputs/<preset>.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional human-readable prefix added to the dated output folder.")
    parser.add_argument("--animate", action="store_true", help="Save an animation for stored trial trajectories.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging to warnings on the console.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed diagnostics in the console and run.log.")
    parser.add_argument("--log-every", type=int, default=None, help="Print progress every N time steps. Default: 25.")
    parser.add_argument(
        "--controller",
        choices=("trbfgs", "gradient"),
        default=None,
        help="Motion optimizer. 'trbfgs' matches the paper logic; 'gradient' is faster for teaching runs.",
    )
    parser.add_argument(
        "--motion-update-every",
        type=int,
        default=None,
        help="Recompute the active-sensing motion direction every N steps; reuse it in between.",
    )
    parser.add_argument(
        "--no-save-traj",
        action="store_true",
        help="Do not save per-step trial histories. This reduces disk use and plotting overhead.",
    )
    parser.add_argument(
        "--save-exact-fim",
        action="store_true",
        help="Save exact posterior-FIM candidate-location maps for selected steps.",
    )
    parser.add_argument(
        "--exact-fim-trials",
        type=int,
        default=None,
        help="Number of initial trials for which exact FIM snapshots are saved. Default: 1.",
    )
    parser.add_argument(
        "--exact-fim-steps",
        type=str,
        default=None,
        help="Comma-separated one-based steps for exact FIM snapshots, e.g. '1,30,60,110,last'.",
    )
    parser.add_argument(
        "--exact-fim-grid",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        default=None,
        help="Candidate grid resolution for exact FIM maps. Default: 95 70.",
    )
    return parser.parse_args(argv)


def make_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig()

    if args.preset == "demo":
        cfg.N_trials = 1
        cfg.steps = 60
        cfg.M = 120
        cfg.S = 8
        cfg.P_field = 80
        cfg.save_trial_trajs = 1
    elif args.preset == "paper":
        cfg.N_trials = 50
        cfg.steps = 400
        cfg.M = 250
        cfg.S = 10
        cfg.P_field = 200
        cfg.save_trial_trajs = 1
    else:
        # Deterministic reference setup useful for quick comparisons and exact-FIM snapshots.
        cfg.N_trials = 1
        cfg.steps = 300
        cfg.base_seed = 123
        cfg.dt = 5.0
        cfg.diffusivity = 1500.0
        cfg.sigma_w_true = 1e-4
        cfg.sigma_w_filter = 2e-2
        cfg.S = 10
        cfg.r_true_val = 1e-6
        cfg.r_filter_val = 1e-4
        cfg.rho_true_mode = "given"
        cfg.rho_true_given = (1500.0, 1500.0)
        cfg.u_true = 250.0
        cfg.M = 250
        cfg.u_min = 1.0
        cfg.u_max = 400.0
        cfg.sigma_rho_init = 25.0
        cfg.sigma_logu = 0.06
        cfg.resample_frac = 0.4
        cfg.use_rho0_init = True
        cfg.rho0_mode = "given"
        cfg.rho0_given = (4500.0, 1000.0)
        cfg.sigma_rho0 = 1500.0
        cfg.u0_mu = 200.0
        cfg.u0_sigma = 200.0
        cfg.delta = 5e-4
        cfg.k_rep_agents = 2e5
        cfg.d0_bbox = 200.0
        cfg.k_rep_bbox = 1e5 
        cfg.dt_move = 14.0          # was 7.0
        cfg.alpha_attr = 2.4        # was 1.4
        cfg.max_speed = 750.0       # was 300.0
        cfg.max_step_per_iter = 550.0  # was 250.0
        cfg.beta_slew = 0.0        # was 5e-6
        cfg.spread_beta = 0.03      # was 0.15
        cfg.d0_agents = 220.0       # was 350.0
        cfg.tr_Delta0 = 100.0
        cfg.tr_Delta_min = 20.0
        cfg.tr_Delta_max = 200.0
        cfg.tr_hess_damp = 0.05
        cfg.tr_max_inner_tries = 2
        cfg.spread_dref = 600.0
        cfg.grad_ema_alpha = 0.20
        cfg.grad_knn_k = 3
        cfg.grad_knn_lam = 0.15
        cfg.P_field = 200
        cfg.field_error_every = 1
        cfg.field_error_metric = "relative"
        cfg.field_wgt_eps = 1e-12
        cfg.save_trial_trajs = 1
        cfg.anim_fps = 6

    if args.trials is not None:
        cfg.N_trials = args.trials
    if args.steps is not None:
        cfg.steps = args.steps
    if args.particles is not None:
        cfg.M = args.particles
    if args.sensors is not None:
        cfg.S = args.sensors
    if args.seed is not None:
        cfg.base_seed = args.seed
    if args.log_every is not None:
        cfg.log_every = max(1, int(args.log_every))
    if args.controller is not None:
        cfg.controller = str(args.controller)
    if args.motion_update_every is not None:
        cfg.motion_update_every = max(1, int(args.motion_update_every))
    if args.no_save_traj:
        cfg.save_trial_trajs = 0
    cfg.debug_logs = bool(args.debug)
    cfg.save_exact_fim = bool(args.save_exact_fim)
    if args.exact_fim_trials is not None:
        cfg.exact_fim_trials = max(1, int(args.exact_fim_trials))
    if args.exact_fim_steps is not None:
        cfg.exact_fim_steps = str(args.exact_fim_steps)
    if args.exact_fim_grid is not None:
        cfg.exact_fim_nx = max(2, int(args.exact_fim_grid[0]))
        cfg.exact_fim_ny = max(2, int(args.exact_fim_grid[1]))

    if not args.animate:
        cfg.anim_fps = 0

    return cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = make_config(args)

    preset_label = args.preset

    outdir = Path(args.output) if args.output else ROOT / "outputs" / preset_label
    outdir.mkdir(parents=True, exist_ok=True)

    run_tag, cfg_tag = build_run_tags(preset_label, cfg, args.run_name)
    cfg_out = outdir / run_tag
    cfg_out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg_out / "run.log", quiet=args.quiet, debug=args.debug)

    log.info("Output folder: %s", cfg_out)
    with open(cfg_out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"preset": preset_label, "run_tag": run_tag, "config": asdict(cfg)}, f, indent=2)

    msh_path = ROOT / cfg.mesh_path
    loaded = load_gmsh_tri(str(msh_path))
    mesh = loaded.mesh
    boundaries = loaded.boundaries
    finder = mesh.element_finder()

    log.info("Mesh nodes=%d elements=%d", mesh.p.shape[1], mesh.t.shape[1])
    log.info(
        "Preset=%s | trials=%d | steps=%d | sensors=%d | particles=%d | controller=%s | motion_update_every=%d | log_every=%d",
        preset_label, cfg.N_trials, cfg.steps, cfg.S, cfg.M, cfg.controller, cfg.motion_update_every, cfg.log_every,
    )
    if cfg.save_exact_fim:
        log.info(
            "Exact FIM snapshots enabled | trials=%d | steps=%s | grid=%dx%d",
            cfg.exact_fim_trials, cfg.exact_fim_steps, cfg.exact_fim_nx, cfg.exact_fim_ny,
        )
    if args.debug:
        log.debug("Debug diagnostics enabled.")

    trial_dir = cfg_out / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    t_asm = time.time()
    model = assemble_advection_diffusion(mesh, boundaries, diffusivity=cfg.diffusivity)
    stepper_true = time_discretize_implicit_euler(
        model, dt=cfg.dt, dirichlet_tag=cfg.dirichlet_tag, sigma_w=cfg.sigma_w_true
    )
    stepper_filt = time_discretize_implicit_euler(
        model, dt=cfg.dt, dirichlet_tag=cfg.dirichlet_tag, sigma_w=cfg.sigma_w_filter
    )
    log.info("Assembled FEM model in %.2f s", time.time() - t_asm)

    rng_fp = np.random.default_rng(cfg.base_seed + 999)
    field_points = np.zeros((cfg.P_field, 2))
    for i in range(cfg.P_field):
        field_points[i] = sample_point_in_domain(mesh, finder, rng_fp)

    field_sampler_true = P1PointSampler(mesh, finder, stepper_true, field_points)
    field_sampler_filt = P1PointSampler(mesh, finder, stepper_filt, field_points)

    map_pos_err_all = np.zeros((cfg.N_trials, cfg.steps))
    mmse_pos_err_all = np.zeros((cfg.N_trials, cfg.steps))
    map_u_err_all = np.zeros((cfg.N_trials, cfg.steps))
    mmse_u_err_all = np.zeros((cfg.N_trials, cfg.steps))
    map_field_rmse_all = np.full((cfg.N_trials, cfg.steps), np.nan)
    mmse_field_rmse_all = np.full((cfg.N_trials, cfg.steps), np.nan)
    traj_samples: List[Dict[str, Any]] = []

    for trial in range(cfg.N_trials):
        seed = cfg.base_seed + trial
        log.info("Trial %d/%d | seed=%d", trial + 1, cfg.N_trials, seed)
        out = run_one_trial(
            mesh=mesh,
            boundaries=boundaries,
            finder=finder,
            stepper_true=stepper_true,
            stepper_filt=stepper_filt,
            cfg=cfg,
            seed=seed,
            field_points=field_points,
            field_sampler_true=field_sampler_true,
            field_sampler_filt=field_sampler_filt,
            trial_index=trial,
            trial_dir=trial_dir,
        )

        map_pos_err_all[trial] = out["map_pos_err"]
        mmse_pos_err_all[trial] = out["mmse_pos_err"]
        map_u_err_all[trial] = out["map_u_err"]
        mmse_u_err_all[trial] = out["mmse_u_err"]
        map_field_rmse_all[trial] = np.sqrt(out["map_field_mse"])
        mmse_field_rmse_all[trial] = np.sqrt(out["mmse_field_mse"])
        if "traj" in out:
            traj_samples.append(out["traj"])

    map_pos_rmse = rmse_over_trials(map_pos_err_all, axis=0)
    mmse_pos_rmse = rmse_over_trials(mmse_pos_err_all, axis=0)
    map_u_rmse = rmse_over_trials(map_u_err_all, axis=0)
    mmse_u_rmse = rmse_over_trials(mmse_u_err_all, axis=0)
    map_field_mean = np.nanmean(map_field_rmse_all, axis=0)
    mmse_field_mean = np.nanmean(mmse_field_rmse_all, axis=0)

    q_map_pos = compute_quantiles(map_pos_err_all, axis=0)
    q_mmse_pos = compute_quantiles(mmse_pos_err_all, axis=0)
    q_map_u = compute_quantiles(map_u_err_all, axis=0)
    q_mmse_u = compute_quantiles(mmse_u_err_all, axis=0)
    q_map_field = compute_quantiles(map_field_rmse_all, axis=0)
    q_mmse_field = compute_quantiles(mmse_field_rmse_all, axis=0)

    np.savez(
        cfg_out / "mc_results.npz",
        cfg=asdict(cfg),
        map_pos_err_all=map_pos_err_all,
        mmse_pos_err_all=mmse_pos_err_all,
        map_u_err_all=map_u_err_all,
        mmse_u_err_all=mmse_u_err_all,
        map_field_rmse_all=map_field_rmse_all,
        mmse_field_rmse_all=mmse_field_rmse_all,
        map_pos_rmse=map_pos_rmse,
        mmse_pos_rmse=mmse_pos_rmse,
        map_u_rmse=map_u_rmse,
        mmse_u_rmse=mmse_u_rmse,
        map_field_mean=map_field_mean,
        mmse_field_mean=mmse_field_mean,
        traj_samples=traj_samples,
    )
    log.info("Saved results: %s", cfg_out / "mc_results.npz")

    plot_with_bands(np.mean(map_pos_err_all, axis=0), q_map_pos,
                    f"{cfg_tag} | Position error (MAP)", "|rho_MAP - rho_true| [m]",
                    cfg_out / "pos_err_map_bands.png", n_trials=cfg.N_trials)
    plot_with_bands(np.mean(mmse_pos_err_all, axis=0), q_mmse_pos,
                    f"{cfg_tag} | Position error (MMSE)", "|rho_MMSE - rho_true| [m]",
                    cfg_out / "pos_err_mmse_bands.png", n_trials=cfg.N_trials)
    plot_with_bands(np.mean(map_u_err_all, axis=0), q_map_u,
                    f"{cfg_tag} | Intensity error (MAP)", "|u_MAP - u_true|",
                    cfg_out / "u_err_map_bands.png", n_trials=cfg.N_trials)
    plot_with_bands(np.mean(mmse_u_err_all, axis=0), q_mmse_u,
                    f"{cfg_tag} | Intensity error (MMSE)", "|u_MMSE - u_true|",
                    cfg_out / "u_err_mmse_bands.png", n_trials=cfg.N_trials)
    field_ylabel = "Field RMSE" if cfg.field_error_metric != "relative" else "NRMSE(field)"
    plot_with_bands(np.nanmean(map_field_rmse_all, axis=0), q_map_field,
                    f"{cfg_tag} | Field error (MAP)", field_ylabel,
                    cfg_out / "field_rmse_map_bands.png", n_trials=cfg.N_trials)
    plot_with_bands(np.nanmean(mmse_field_rmse_all, axis=0), q_mmse_field,
                    f"{cfg_tag} | Field error (MMSE)", field_ylabel,
                    cfg_out / "field_rmse_mmse_bands.png", n_trials=cfg.N_trials)

    k = np.arange(cfg.steps) + 1
    plt.figure()
    plt.plot(k, map_pos_rmse, label="MAP")
    plt.plot(k, mmse_pos_rmse, label="MMSE")
    plt.grid(True)
    plt.xlabel("step k")
    plt.ylabel("position RMSE [m]")
    plt.title(f"{cfg_tag} | Source position RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg_out / "rmse_position.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(k, map_u_rmse, label="MAP")
    plt.plot(k, mmse_u_rmse, label="MMSE")
    plt.grid(True)
    plt.xlabel("step k")
    plt.ylabel("intensity RMSE")
    plt.title(f"{cfg_tag} | Source intensity RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg_out / "rmse_intensity.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(k, map_field_mean, label="MAP")
    plt.plot(k, mmse_field_mean, label="MMSE")
    plt.grid(True)
    plt.xlabel("step k")
    plt.ylabel(field_ylabel)
    plt.title(f"{cfg_tag} | Field estimation error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg_out / "field_error.png", dpi=200)
    plt.close()

    if traj_samples:
        ts0 = traj_samples[0]
        plt.figure()
        plt.plot(ts0["rho_map"][:, 0], ts0["rho_map"][:, 1], label="MAP")
        plt.plot(ts0["rho_mmse"][:, 0], ts0["rho_mmse"][:, 1], label="MMSE")
        plt.scatter([ts0["rho_true"][0]], [ts0["rho_true"][1]], marker="*", s=120, label="true source")
        plt.axis("equal")
        plt.grid(True)
        plt.title(f"{cfg_tag} | Estimated source trajectory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg_out / "sample_source_trajectory.png", dpi=200)
        plt.close()

    if args.animate:
        generate_animations_for_saved_trials(mesh=mesh, trial_dir=trial_dir, out_dir=cfg_out, fps=cfg.anim_fps)

    log.info("Done. Outputs in: %s", cfg_out)


if __name__ == "__main__":
    main()
