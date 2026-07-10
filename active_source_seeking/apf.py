# src/apf.py
from __future__ import annotations

import numpy as np


def inside_mesh(mesh, finder, p) -> bool:
    try:
        e = finder(np.array([p[0]]), np.array([p[1]]))
        e = int(np.atleast_1d(e)[0])
        return e >= 0
    except ValueError:
        return False

def sample_point_in_domain(mesh, finder, rng: np.random.Generator, max_tries: int = 20000) -> np.ndarray:
    xmin, xmax = mesh.p[0].min(), mesh.p[0].max()
    ymin, ymax = mesh.p[1].min(), mesh.p[1].max()
    for _ in range(max_tries):
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float)
        if inside_mesh(mesh, finder, p):
            return p
    raise RuntimeError("Could not sample interior point.")

def backtracking_project_inside(
    mesh,
    finder,
    p0: np.ndarray,
    dp: np.ndarray,
    rng: np.random.Generator | None = None,
    max_tries: int = 12,
    jitter_tries: int = 6,
    jitter_scale: float = 0.25,
) -> np.ndarray:
    """
    Try p = p0 + dp; if outside mesh, shrink dp (halve) until inside.
    If that fails, try a few jittered directions (still shrinking).
    If still failing and rng provided, resample an interior point.
    """
    p = p0 + dp
    if inside_mesh(mesh, finder, p):
        return p

    # 1) pure backtracking
    alpha = 1.0
    for _ in range(max_tries):
        alpha *= 0.5
        p = p0 + alpha * dp
        if inside_mesh(mesh, finder, p):
            return p

    # 2) jittered directions (helps near concave boundaries / holes)
    if rng is not None:
        base = dp.copy()
        nb = float(np.linalg.norm(base) + 1e-12)
        # perpendicular unit
        perp = np.array([-base[1], base[0]], dtype=float) / nb

        for _ in range(jitter_tries):
            # random mix of forward + sideways
            side = (rng.uniform(-1.0, 1.0) * jitter_scale) * nb
            dp2 = base + side * perp

            alpha = 1.0
            for _ in range(max_tries):
                alpha *= 0.5
                p = p0 + alpha * dp2
                if inside_mesh(mesh, finder, p):
                    return p

        return sample_point_in_domain(mesh, finder, rng)

    return p0


def pairwise_repulsion(sensors: np.ndarray, i: int, d0: float, k_rep: float) -> np.ndarray:
    """
    Repulsive velocity from other agents:
      v = sum_j k_rep * (1/d - 1/d0) * (1/d^3) * (s_i - s_j)   for d < d0
    """
    si = sensors[i]
    v = np.zeros(2, dtype=float)
    for j in range(sensors.shape[0]):
        if j == i:
            continue
        dvec = si - sensors[j]
        d = float(np.linalg.norm(dvec) + 1e-12)
        if d < d0:
            v += k_rep * (1.0 / d - 1.0 / d0) * (1.0 / (d**3)) * dvec
    return v


def boundary_repulsion_bbox(mesh, p: np.ndarray, d0: float, k_rep: float) -> np.ndarray:
    """
    Cheap repulsion from the mesh bounding box (not true coastline/island distance,
    but good enough to stop sensors drifting outside).
    Later you can replace with signed-distance-to-boundary.
    """
    xmin, xmax = float(mesh.p[0].min()), float(mesh.p[0].max())
    ymin, ymax = float(mesh.p[1].min()), float(mesh.p[1].max())

    x, y = float(p[0]), float(p[1])
    v = np.zeros(2, dtype=float)

    # distance to each wall
    dl = x - xmin
    dr = xmax - x
    db = y - ymin
    dt = ymax - y

    def repulse(d, direction):
        # direction is +- unit vector
        if d < d0:
            return k_rep * (1.0 / (d + 1e-12) - 1.0 / d0) * (1.0 / ((d + 1e-12) ** 2)) * direction
        return 0.0 * direction

    v += repulse(dl, np.array([+1.0, 0.0]))
    v += repulse(dr, np.array([-1.0, 0.0]))
    v += repulse(db, np.array([0.0, +1.0]))
    v += repulse(dt, np.array([0.0, -1.0]))
    return v


def apf_step(
    mesh,
    finder,
    sensors: np.ndarray,
    grad_dirs: np.ndarray,
    dt_move: float,
    alpha_attr: float,
    d0_agents: float,
    k_rep_agents: float,
    d0_bbox: float,
    k_rep_bbox: float,
    max_speed: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    One APF-constrained motion step:
      v_attr = alpha * normalized(grad_dir)
      v_rep  = boundary repulsion + pairwise repulsion
      step = dt_move * (v_attr + v_rep)
      then backtracking projection inside mesh
    """
    S = sensors.shape[0]
    new_s = sensors.copy()

    for i in range(S):
        g = grad_dirs[i].astype(float)
        ng = float(np.linalg.norm(g))
        v_attr = alpha_attr * (g / (ng + 1e-12))

        v_rep = np.zeros(2, dtype=float)
        v_rep += boundary_repulsion_bbox(mesh, sensors[i], d0=d0_bbox, k_rep=k_rep_bbox)
        v_rep += pairwise_repulsion(sensors, i=i, d0=d0_agents, k_rep=k_rep_agents)

        # damp attraction when repulsion is strong
        rep_norm = float(np.linalg.norm(v_rep))
        v_attr *= 1.0 / (1.0 + 0.002 * rep_norm)

        v = v_attr + v_rep

        nv = float(np.linalg.norm(v))
        if nv > max_speed:
            v = v * (max_speed / (nv + 1e-12))

        dp = dt_move * v
        new_s[i] = backtracking_project_inside(mesh, finder, sensors[i], dp, rng=rng)

    return new_s
