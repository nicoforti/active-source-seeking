from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from typing import Optional, Tuple

from active_source_seeking.source_model import source_injection_vector
from active_source_seeking.apf import inside_mesh
from active_source_seeking.pf_utils import uncertainty_radius_2sigma

EPS = 1e-12


def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m)) + 1e-300))


def _ess(w: np.ndarray) -> float:
    return float(1.0 / (np.sum(w * w) + 1e-18))


def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    M = len(w)
    positions = (rng.random() + np.arange(M)) / M
    cdf = np.cumsum(w)
    return np.searchsorted(cdf, positions)


def _predict_information_free(Omega_prev: np.ndarray, A_free: np.ndarray, Qinv_free: sp.spmatrix) -> np.ndarray:
    """
    Omega_pred = Q^{-1} - Q^{-1} A (Omega_prev + A^T Q^{-1} A)^{-1} A^T Q^{-1}
    The implementation assumes a diagonal process information matrix.
    """
    qinv = Qinv_free.diagonal().astype(float)
    QinvA = qinv[:, None] * A_free
    AtQinvA = A_free.T @ QinvA
    S = 0.5 * (Omega_prev + AtQinvA + (Omega_prev + AtQinvA).T)
    RHS = A_free.T * qinv[None, :]
    cF = la.cho_factor(S + 1e-12 * np.eye(S.shape[0]), lower=True, check_finite=False)
    X = la.cho_solve(cF, RHS, check_finite=False)
    Omega_pred = np.diag(qinv) - (qinv[:, None] * (A_free @ X))
    return 0.5 * (Omega_pred + Omega_pred.T)


def _compute_Psi(Lambda: np.ndarray, Omega_k: np.ndarray) -> np.ndarray:
    cF = la.cho_factor(Omega_k + 1e-12 * np.eye(Omega_k.shape[0]), lower=True, check_finite=False)
    X = la.cho_solve(cF, Lambda, check_finite=False)
    return Lambda - Lambda @ X


def fe_impf_step_free(
    mesh,
    stepper,
    dt: float,
    Omega_prev: np.ndarray,
    q_prev: np.ndarray,
    z_prev: np.ndarray,
    w_prev: np.ndarray,
    Lambda_sp: sp.csr_matrix,
    lam: np.ndarray,
    bbar_free: np.ndarray,
    rng: np.random.Generator,
    finder=None,
    resample_threshold: Optional[float] = None,
    # proposals
    sigma_rho: float = 25.0,
    sigma_logu: float = 0.10,
    # bounds
    u_min: float = 1.0,
    u_max: float = 500.0,
    # prior on u (log-space) scheduled by rho uncertainty
    sigma_logu_prior_min: float = 0.20,
    sigma_logu_prior_max: float = 1.00,
    prior_r0: float = 2000.0,
    u_prior_follow: float = 0.05,
    u0_mu: float = 200.0,
    # behavior
    use_u_marginal_weights: bool = True,
    sample_u_when_weak: bool = True,
    weak_like_rel_std: float = 0.25,       # sample u if std(u|j) > rel*mu
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    FE-IMPF on free dofs with particles z=[rho_x, rho_y, u].
    - rho: RW + backtracking projection into mesh
    - u: Rao–Blackwellized per-particle (Gaussian approx) with non-locking prior anchor (uses u0_mu)
    - weights: based on correlated-innovation Psi. Optionally marginalize u into weights.
    """

    M, nfree = q_prev.shape

    if finder is None:
        finder = mesh.element_finder()

    solve_free = stepper.solve_free
    restrict_full = stepper.restrict_full

    A_free = stepper.A_free.toarray() if sp.issparse(stepper.A_free) else stepper.A_free

    # ----- dense Lambda -----
    Lambda = Lambda_sp.toarray()
    Lambda = 0.5 * (Lambda + Lambda.T)

    # ----- (1) propose rho -----
    z = z_prev.copy()
    dp_all = sigma_rho * rng.standard_normal(size=(M, 2))
    for j in range(M):
        p0 = z_prev[j, 0:2]
        p1 = p0 + dp_all[j]

        if inside_mesh(mesh, finder, p1):
            z[j, 0:2] = p1
            continue

        # backtracking toward p0
        dp = p1 - p0
        alpha = 0.5
        ok = False
        for _ in range(12):
            pc = p0 + alpha * dp
            if inside_mesh(mesh, finder, pc):
                z[j, 0:2] = pc
                ok = True
                break
            alpha *= 0.5
        if not ok:
            z[j, 0:2] = p0

    # Initialize u before the Rao-Blackwellized update.
    z[:, 2] = np.clip(z_prev[:, 2], u_min, u_max)

    # ----- (2) predict Omega -----
    Omega_pred = _predict_information_free(Omega_prev, A_free, stepper.Qinv_free)

    # ----- (3) compute Omega_k and Psi -----
    Omega_k = Omega_pred + Lambda
    Omega_k = 0.5 * (Omega_k + Omega_k.T) + 1e-12 * np.eye(nfree)
    Psi = _compute_Psi(Lambda, Omega_k)
    Psi = 0.5 * (Psi + Psi.T)

    # ----- (4) recover x_prev and deterministic prediction -----
    cF_prev = la.cho_factor(Omega_prev + 1e-12 * np.eye(Omega_prev.shape[0]), lower=True, check_finite=False)
    x_prev = la.cho_solve(cF_prev, q_prev.T, check_finite=False).T
    x_det = (A_free @ x_prev.T).T

    # ----- (5) prior schedule based on rho uncertainty -----
    try:
        rad_2sig = float(uncertainty_radius_2sigma(z_prev, w_prev))
    except Exception:
        rad_2sig = float(prior_r0)

    t = float(np.clip(rad_2sig / (prior_r0 + EPS), 0.0, 1.0))
    sigma_logu_prior = float(sigma_logu_prior_min + t * (sigma_logu_prior_max - sigma_logu_prior_min))
    var_logu_prior = float(sigma_logu_prior * sigma_logu_prior)

    u_center = float(np.clip(u0_mu, u_min, u_max))
    logu_center = float(np.log(u_center))

    u_mmse_prev = float(np.sum(w_prev * z_prev[:, 2]))
    logu_prev = float(np.log(np.clip(u_mmse_prev, u_min, u_max)))

    alpha_follow_prev = float(np.clip(1.0 - t, 0.0, 1.0))  # rad large => small follow
    logu_prior_global = (1.0 - alpha_follow_prev) * logu_center + alpha_follow_prev * logu_prev

    # ----- (6) RB update for u and build x_pred -----
    x_pred = np.empty_like(x_det)
    u_post_mean = np.empty(M, dtype=float)
    u_post_var = np.empty(M, dtype=float)

    # optional extra term for marginal weights
    logl_u_marg = np.zeros(M, dtype=float)
    logl_marg = np.empty(M, dtype=float)   # log p(y | rho^j) (u marginalizzato)

    for j in range(M):
        rho = z[j, 0:2]

        b_full, _, _ = source_injection_vector(mesh, rho, finder=finder)
        b_free = restrict_full(b_full)

        x0 = x_det[j] + bbar_free
        x1 = solve_free(dt * b_free)

        # per-particle prior mean in log-space (mild follow of particle value)
        logu_prior_particle = float(np.log(np.clip(z_prev[j, 2], u_min, u_max)))
        logu_prior = (1.0 - u_prior_follow) * logu_prior_global + u_prior_follow * logu_prior_particle

        u0 = float(np.exp(logu_prior))
        u0 = float(np.clip(u0, u_min, u_max))
        var_u_prior = float((u0 * u0) * var_logu_prior + EPS)  # delta method

        Px1 = Psi @ x1
        Px0 = Psi @ x0

        # Termini "likelihood" (dipendono da rho via x0,x1)
        a_like = float(x1 @ Px1)                  # x1^T Psi x1
        b_like = float(x1 @ (lam - Px0))          # x1^T (lam - Psi x0)

        # Costante in rho (la parte di x0 nella likelihood)
        const_rho = float(x0 @ lam - 0.5 * (x0 @ Px0))

        # Prior su u: N(u0, var_u_prior) (in u-space, delta-method)
        a = a_like + 1.0 / var_u_prior
        b = b_like + u0 / var_u_prior

        # RB mean/var per costruire x_pred (come già fai)
        if a < 1e-12:
            mu_u = u0
            var_u = var_u_prior
        else:
            mu_u = b / a
            var_u = 1.0 / a

        mu_u = float(np.clip(mu_u, u_min, u_max))
        var_u = float(max(var_u, 1e-18))

        # sample (opzionale) quando informazione debole
        if sample_u_when_weak:
            std_u = float(np.sqrt(var_u))
            if std_u > weak_like_rel_std * max(mu_u, 1.0):
                u_s = mu_u + std_u * rng.standard_normal()
                z[j, 2] = float(np.clip(u_s, u_min, u_max))
            else:
                z[j, 2] = mu_u
        else:
            z[j, 2] = mu_u

        x_pred[j] = x0 + z[j, 2] * x1

        u_post_mean[j] = mu_u
        u_post_var[j]  = var_u

        # ---- PESI marginalizzati su u (CORRETTO) ----
        # Integrale di exp( const_rho - 0.5*a_like u^2 + b_like u  -0.5*(u-u0)^2/var )
        # => const_rho - 0.5*(u0^2/var) + 0.5*b^2/a - 0.5*log(a)  (a,b includono prior)
        logl_marg[j] = const_rho - 0.5 * (u0 * u0) / var_u_prior + 0.5 * (b * b) / (a + EPS) - 0.5 * np.log(a + EPS)

        if use_u_marginal_weights:
            # log p(y|rho^j) contribution after integrating u (up to constants):
            # 0.5*b^2/a - 0.5*log(a)
            logl_u_marg[j] = 0.5 * (b * b) / (a + EPS) - 0.5 * np.log(a + EPS)

    # ----- (7) weights -----
    v = x_pred @ lam
    Px = (Psi @ x_pred.T).T
    quad = np.einsum("ij,ij->i", x_pred, Px)
    logl = v - 0.5 * quad

    if use_u_marginal_weights:
        logl = logl_marg
    else:
        # fallback: likelihood valutata a u campionato (meno stabile)
        v = x_pred @ lam
        Px = (Psi @ x_pred.T).T
        quad = np.einsum("ij,ij->i", x_pred, Px)
        logl = v - 0.5 * quad

    logw = np.log(w_prev + 1e-300) + logl
    logw -= _logsumexp(logw)
    w = np.exp(logw)

    # ----- (8) resample if needed -----
    if (resample_threshold is not None) and (_ess(w) < float(resample_threshold)):
        idx = _systematic_resample(w, rng)
        z = z[idx]
        x_prev = x_prev[idx]
        w = np.full(M, 1.0 / M)

        # keep the same logu_prior_global computed above.

        # rebuild deterministic prediction after resampling
        x_det = (A_free @ x_prev.T).T

        # rebuild x_pred consistently (RB again, but keep diversity logic)
        for j in range(M):
            rho = z[j, 0:2]
            b_full, _, _ = source_injection_vector(mesh, rho, finder=finder)
            b_free = restrict_full(b_full)

            x0 = x_det[j] + bbar_free
            x1 = solve_free(dt * b_free)

            logu_prior_particle = float(np.log(np.clip(z[j, 2], u_min, u_max)))
            logu_prior = (1.0 - u_prior_follow) * logu_prior_global + u_prior_follow * logu_prior_particle

            u0 = float(np.exp(logu_prior))
            u0 = float(np.clip(u0, u_min, u_max))
            var_u_prior = float((u0 * u0) * var_logu_prior + EPS)

            Px1 = Psi @ x1
            Px0 = Psi @ x0

            a = float(x1 @ Px1 + 1.0 / var_u_prior)
            b = float(x1 @ (lam - Px0) + u0 / var_u_prior)

            if a < 1e-12:
                mu_u = u0
                var_u = var_u_prior
            else:
                mu_u = b / a
                var_u = 1.0 / a

            mu_u = float(np.clip(mu_u, u_min, u_max))
            var_u = float(max(var_u, 1e-18))

            if sample_u_when_weak:
                std_u = float(np.sqrt(var_u))
                if std_u > weak_like_rel_std * max(mu_u, 1.0):
                    u_s = mu_u + std_u * rng.standard_normal()
                    z[j, 2] = float(np.clip(u_s, u_min, u_max))
                else:
                    z[j, 2] = mu_u
            else:
                z[j, 2] = mu_u

            x_pred[j] = x0 + z[j, 2] * x1

    # ----- (9) q update -----
    q_pred = (Omega_pred @ x_pred.T).T
    q = q_pred + lam[None, :]

    return z, w, q, Omega_k, Omega_pred, x_pred, Psi
