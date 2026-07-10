from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.linalg as la

from active_source_seeking.active_sensing import build_measurement_information_free_sparse
from active_source_seeking.central_controller import bayes_fim_free


def parse_snapshot_steps(spec: str | Iterable[int] | None, steps: int) -> set[int]:
    """Parse step indices for exact FIM snapshots.

    Integer tokens are interpreted as one-based display steps, matching paper
    figures such as ``k=1,30,60``. Token ``0`` is also accepted and maps to the
    first internal index. Token ``last`` maps to ``steps - 1``.
    """
    if spec is None:
        return set()
    if isinstance(spec, str):
        tokens = [tok.strip().lower() for tok in spec.split(",") if tok.strip()]
        out: set[int] = set()
        for tok in tokens:
            if tok in {"last", "final", "end"}:
                out.add(max(0, int(steps) - 1))
            else:
                step_num = int(tok)
                out.add(0 if step_num <= 0 else step_num - 1)
        return {k for k in out if 0 <= k < int(steps)}
    return {int(k) for k in spec if 0 <= int(k) < int(steps)}


def _safe_cholesky(Omega: np.ndarray, base_jitter: float = 1e-14, max_tries: int = 12):
    Omega = 0.5 * (np.asarray(Omega, float) + np.asarray(Omega, float).T)
    eye = np.eye(Omega.shape[0])
    jitter = float(base_jitter)
    last_error = None
    for _ in range(max_tries):
        try:
            Omega_j = Omega + jitter * eye
            return la.cho_factor(Omega_j, lower=True, check_finite=False), Omega_j, jitter
        except la.LinAlgError as exc:
            last_error = exc
            jitter *= 10.0
    raise la.LinAlgError(f"Omega_k is not SPD after jitter; last error: {last_error}")


def posterior_fim_for_sensors(
    mesh,
    stepper,
    sensors_xy: np.ndarray,
    y: np.ndarray,
    r_filter: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    Omega_k: np.ndarray,
    finder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the exact posterior expected FIM used by the controller.

    The returned matrix is

        Fbar = sum_j w_j G_j.T @ Psi @ G_j,

    with ``G_j`` assembled from the FE source injection vector and its
    derivatives with respect to ``[rho_x, rho_y, u]``.
    """
    Lambda_sp, _ = build_measurement_information_free_sparse(
        mesh, stepper, sensors_xy, y, r_filter, finder=finder
    )
    Lambda_sp = 0.5 * (Lambda_sp + Lambda_sp.T)
    Lambda = Lambda_sp.toarray()
    Lambda = 0.5 * (Lambda + Lambda.T)

    cF, _, _ = _safe_cholesky(Omega_k)
    X = la.cho_solve(cF, Lambda, check_finite=False)
    Psi = Lambda - Lambda @ X
    Psi = 0.5 * (Psi + Psi.T)

    weights = np.asarray(w, float).ravel()
    weights = weights / (weights.sum() + 1e-15)
    Fbar = bayes_fim_free(mesh, stepper, Psi, np.asarray(z, float), weights, finder=finder)
    Fbar = 0.5 * (Fbar + Fbar.T)
    return Fbar, Psi, Lambda


def fim_scores(Fbar: np.ndarray, delta: float, w_u: float = 1.0) -> tuple[float, float]:
    """Return ``logdet(W Fbar W + delta I)`` and ``trace(W Fbar W)``."""
    W = np.diag([1.0, 1.0, float(w_u)])
    Fw = W @ np.asarray(Fbar, float) @ W
    Fw = 0.5 * (Fw + Fw.T)
    sign, logdet = np.linalg.slogdet(Fw + float(delta) * np.eye(3))
    if sign <= 0 or not np.isfinite(logdet):
        logdet = np.nan
    return float(logdet), float(np.trace(Fw))


def exact_candidate_fim_grid(
    mesh,
    stepper,
    z: np.ndarray,
    w: np.ndarray,
    Omega_k: np.ndarray,
    finder,
    r_value: float,
    delta: float,
    w_u: float = 1.0,
    nx: int = 95,
    ny: int = 70,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save_fbar_grid: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate exact posterior-FIM scores over candidate single-sensor sites."""
    if xlim is None:
        xlim = (float(mesh.p[0].min()), float(mesh.p[0].max()))
    if ylim is None:
        ylim = (float(mesh.p[1].min()), float(mesh.p[1].max()))

    xs = np.linspace(xlim[0], xlim[1], int(nx))
    ys = np.linspace(ylim[0], ylim[1], int(ny))
    info_logdet = np.full((len(ys), len(xs)), np.nan, dtype=float)
    info_trace = np.full_like(info_logdet, np.nan)
    fbar_grid = np.full((len(ys), len(xs), 3, 3), np.nan, dtype=float) if save_fbar_grid else None

    y_dummy = np.zeros(1, dtype=float)
    r_single = np.array([float(r_value)], dtype=float)

    for iy, yy in enumerate(ys):
        for ix, xx in enumerate(xs):
            candidate = np.array([[xx, yy]], dtype=float)
            try:
                finder(np.array([xx]), np.array([yy]))
                Fbar, _, _ = posterior_fim_for_sensors(
                    mesh=mesh,
                    stepper=stepper,
                    sensors_xy=candidate,
                    y=y_dummy,
                    r_filter=r_single,
                    z=z,
                    w=w,
                    Omega_k=Omega_k,
                    finder=finder,
                )
            except Exception:
                continue

            logdet, trace = fim_scores(Fbar, delta=delta, w_u=w_u)
            info_logdet[iy, ix] = logdet
            info_trace[iy, ix] = trace
            if fbar_grid is not None:
                fbar_grid[iy, ix] = Fbar

    out = {
        "xs": xs,
        "ys": ys,
        "info_logdet": info_logdet,
        "info_trace": info_trace,
    }
    if fbar_grid is not None:
        out["fbar_grid"] = fbar_grid
    return out


def save_exact_fim_snapshot(
    out_dir: str | Path,
    k: int,
    mesh,
    stepper,
    sensors_xy: np.ndarray,
    y: np.ndarray,
    r_filter: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    Omega_k: np.ndarray,
    finder,
    delta: float,
    w_u: float = 1.0,
    rho_map: np.ndarray | None = None,
    rho_mmse: np.ndarray | None = None,
    rho_true: np.ndarray | None = None,
    nx: int = 95,
    ny: int = 70,
) -> Path:
    """Save an exact FIM snapshot for the current posterior state."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sensors_xy = np.asarray(sensors_xy, float)
    y = np.asarray(y, float).ravel()
    r_filter = np.asarray(r_filter, float).ravel()
    r_value = float(np.mean(r_filter))

    team_fbar, team_psi, team_lambda = posterior_fim_for_sensors(
        mesh=mesh,
        stepper=stepper,
        sensors_xy=sensors_xy,
        y=y,
        r_filter=r_filter,
        z=z,
        w=w,
        Omega_k=Omega_k,
        finder=finder,
    )
    team_logdet, team_trace = fim_scores(team_fbar, delta=delta, w_u=w_u)

    grid = exact_candidate_fim_grid(
        mesh=mesh,
        stepper=stepper,
        z=z,
        w=w,
        Omega_k=Omega_k,
        finder=finder,
        r_value=r_value,
        delta=delta,
        w_u=w_u,
        nx=nx,
        ny=ny,
        save_fbar_grid=True,
    )

    out_path = out_dir / f"exact_fim_snapshot_k{k:03d}.npz"
    np.savez_compressed(
        out_path,
        k=int(k),
        step=int(k) + 1,
        sensors=sensors_xy,
        y=y,
        r_filter=r_filter,
        rho_map=np.asarray(rho_map, float) if rho_map is not None else np.full(2, np.nan),
        rho_mmse=np.asarray(rho_mmse, float) if rho_mmse is not None else np.full(2, np.nan),
        rho_true=np.asarray(rho_true, float) if rho_true is not None else np.full(2, np.nan),
        team_fbar=team_fbar,
        team_psi=team_psi,
        team_lambda=team_lambda,
        team_logdet=team_logdet,
        team_trace=team_trace,
        delta=float(delta),
        w_u=float(w_u),
        r_value=r_value,
        score_description="Exact posterior FIM: Fbar=sum_j w_j G_j.T @ Psi @ G_j; grid score=logdet(W Fbar W + delta I) and trace(W Fbar W).",
        **grid,
    )
    return out_path
