# src/central_controller.py
from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from active_source_seeking.source_model import source_injection_vector
from active_source_seeking.fem_sampling_p1 import fe_row_and_grad_p1, fe_p1_triplet


def bayes_fim_free(mesh, stepper, Psi: np.ndarray, z: np.ndarray, w: np.ndarray, finder) -> np.ndarray:
    """
    Fbar = sum_j w_j G_j^T Psi G_j, free dofs.
    theta = [rho_x, rho_y, u]
    """
    Fbar = np.zeros((3, 3), dtype=float)
    free = stepper.free
    for j in range(z.shape[0]):
        rho = z[j, 0:2]
        u = float(z[j, 2])
        b, dbdx, dbdy = source_injection_vector(mesh, rho, finder=finder)
        G = np.column_stack([dbdx[free] * u, dbdy[free] * u, b[free]])  # (nfree,3)
        Fbar += float(w[j]) * (G.T @ Psi @ G)
    return Fbar


def aoptimal_loss(Fbar: np.ndarray, delta: float) -> float:
    SL = Fbar + delta * np.eye(Fbar.shape[0])
    return float(np.trace(np.linalg.inv(SL)))


def aoptimal_loss_normalized(Fbar: np.ndarray, delta: float) -> float:
    """
    Normalized A-optimal: (delta/d)*trace(inv(Fbar + delta I)).
    Much less jumpy in magnitude.
    """
    d = Fbar.shape[0]
    SL = Fbar + delta * np.eye(d)
    return float((delta / d) * np.trace(np.linalg.inv(SL)))


def doptimal_loss(Fbar: np.ndarray, delta: float) -> float:
    """
    D-optimal objective: minimize -logdet(Fbar + delta I).
    """
    d = Fbar.shape[0]
    SL = Fbar + delta * np.eye(d)
    sign, logdet = np.linalg.slogdet(SL)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e12
    return float(-logdet)


def doptimal_loss_weighted(Fbar: np.ndarray, delta: float, w_u: float = 1.0) -> float:
    W = np.diag([1.0, 1.0, float(w_u)])
    Fw = W @ Fbar @ W
    SL = Fw + delta * np.eye(3)
    sign, logdet = np.linalg.slogdet(SL)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e12
    return float(-logdet)


def aoptimal_loss_weighted(Fbar: np.ndarray, delta: float, w_u: float = 5.0) -> float:
    """
    Weighted A-optimal: trace(inv(W Fbar W + delta I))
    """
    W = np.diag([1.0, 1.0, float(w_u)])
    SL = W @ Fbar @ W + delta * np.eye(3)
    return float(np.trace(np.linalg.inv(SL)))

def grad_loss_one_sensor_free(
    mesh,
    stepper,
    sensors_xy: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    Omega: np.ndarray,
    Lambda_sp: sp.csr_matrix,
    lam: np.ndarray,
    delta: float,
    i: int,
    finder,
    cF=None,
    X=None,
    Psi=None,
    objective: str = "dopt",   # "dopt" or "aopt"
    w_u: float = 5.0,
) -> np.ndarray:
    """
    Gradient wrt sensor i position (x,y), free dofs.

    objective:
      - "dopt": L = -logdet(W Fbar W + delta I)    -> dL = -tr(SL^{-1} dSL)
      - "aopt": L = tr((W Fbar W + delta I)^{-1})  -> dL = -tr(SL^{-1} dSL SL^{-1})

    Notes:
      - Uses correlated-innovation Psi = Lambda - Lambda Omega^{-1} Lambda
      - Uses weighted FIM: Fw = W Fbar W with W = diag(1,1,w_u)
      - Robust casting of w_u (avoids "float(array)" crashes)
    """
    try:
        w_u = float(np.asarray(w_u).reshape(()))
    except Exception:
        w_u = 1.0
    if not np.isfinite(w_u) or w_u <= 0.0:
        w_u = 1.0

    nfree = int(stepper.free.shape[0])

    # global->free map
    g2f = -np.ones(stepper.N, dtype=int)
    g2f[stepper.free] = np.arange(nfree, dtype=int)

    # sensor inside?
    p = sensors_xy[i]
    try:
        _ = finder(np.array([p[0]]), np.array([p[1]]))
    except Exception:
        return np.zeros(2, dtype=float)

    # Cholesky of Omega
    if cF is None:
        cF = la.cho_factor(Omega, lower=True, check_finite=False)

    # symmetrize Lambda
    Lambda_sp = 0.5 * (Lambda_sp + Lambda_sp.T)
    Lambda = Lambda_sp.toarray()
    Lambda = 0.5 * (Lambda + Lambda.T)

    # caches: X = Omega^{-1} Lambda, Psi = Lambda - Lambda X
    if X is None:
        X = la.cho_solve(cF, Lambda, check_finite=False)
    if Psi is None:
        Psi = Lambda - Lambda @ X
    Psi = 0.5 * (Psi + Psi.T)

    # build Fbar
    Fbar = np.zeros((3, 3), dtype=float)
    free = stepper.free
    for j in range(z.shape[0]):
        rho = z[j, 0:2]
        u = float(z[j, 2])
        b, dbdx, dbdy = source_injection_vector(mesh, rho, finder=finder)
        G = np.column_stack([dbdx[free] * u, dbdy[free] * u, b[free]])  # (nfree,3)
        Fbar += float(w[j]) * (G.T @ Psi @ G)
    Fbar = 0.5 * (Fbar + Fbar.T)

    # weighted FIM and objective matrix
    W = np.diag([1.0, 1.0, w_u])
    Fw = W @ Fbar @ W
    SL = Fw + float(delta) * np.eye(3)

    # stable inverse (3x3) + guard
    sign, logdet = np.linalg.slogdet(SL)
    if sign <= 0 or (not np.isfinite(logdet)):
        return np.zeros(2, dtype=float)
    SLinv = np.linalg.inv(SL)

    # P1 row + gradients at sensor location
    try:
        tri_nodes, _, _ = fe_p1_triplet(mesh, p, finder=finder)
        c_full, cx_full, cy_full, _ = fe_row_and_grad_p1(mesh, p, finder=finder)
    except Exception:
        return np.zeros(2, dtype=float)

    tri_nodes = np.asarray(tri_nodes, dtype=int).ravel()[:3]
    c_loc = c_full[tri_nodes]
    cx_loc = cx_full[tri_nodes]
    cy_loc = cy_full[tri_nodes]

    fidx = g2f[tri_nodes]
    ri = float(r[i])
    if (not np.isfinite(ri)) or ri <= 0.0:
        return np.zeros(2, dtype=float)
    wi = 1.0 / ri

    valid = [(a, int(fidx[a])) for a in range(3) if int(fidx[a]) >= 0]
    if len(valid) == 0:
        return np.zeros(2, dtype=float)

    # dLambda/dx, dLambda/dy on free dofs (dense here)
    rows, cols, data_x, data_y = [], [], [], []
    for a, ia in valid:
        for b, ib in valid:
            rows.append(ia)
            cols.append(ib)
            data_x.append(wi * float(cx_loc[a] * c_loc[b] + c_loc[a] * cx_loc[b]))
            data_y.append(wi * float(cy_loc[a] * c_loc[b] + c_loc[a] * cy_loc[b]))
    dLam_x = sp.coo_matrix((data_x, (rows, cols)), shape=(nfree, nfree)).toarray()
    dLam_y = sp.coo_matrix((data_y, (rows, cols)), shape=(nfree, nfree)).toarray()
    dLam_x = 0.5 * (dLam_x + dLam_x.T)
    dLam_y = 0.5 * (dLam_y + dLam_y.T)

    # Psi = Lambda - Lambda Omega^{-1} Lambda, with Omega = Omega_pred + Lambda.
    # Hence dOmega = dLambda and
    # dPsi = dLambda - dLambda Omega^{-1} Lambda - Lambda Omega^{-1} dLambda
    #        + Lambda Omega^{-1} dLambda Omega^{-1} Lambda.
    def dPsi_from_dLam(dLam_dense: np.ndarray) -> np.ndarray:
        Y = la.cho_solve(cF, dLam_dense, check_finite=False)  # Omega^{-1} dLambda
        dPsi = dLam_dense - dLam_dense @ X - Lambda @ Y + Lambda @ Y @ X
        return 0.5 * (dPsi + dPsi.T)

    dPsix = dPsi_from_dLam(dLam_x)
    dPsiy = dPsi_from_dLam(dLam_y)

    # dFbar = sum_j w_j G^T dPsi G
    dFbar_x = np.zeros((3, 3), dtype=float)
    dFbar_y = np.zeros((3, 3), dtype=float)
    for j in range(z.shape[0]):
        rho = z[j, 0:2]
        u = float(z[j, 2])
        b, dbdx, dbdy = source_injection_vector(mesh, rho, finder=finder)
        G = np.column_stack([dbdx[free] * u, dbdy[free] * u, b[free]])
        dFbar_x += float(w[j]) * (G.T @ dPsix @ G)
        dFbar_y += float(w[j]) * (G.T @ dPsiy @ G)
    dFbar_x = 0.5 * (dFbar_x + dFbar_x.T)
    dFbar_y = 0.5 * (dFbar_y + dFbar_y.T)

    # weighted differential: d(W F W) = W dF W
    dSL_x = W @ dFbar_x @ W
    dSL_y = W @ dFbar_y @ W

    obj = objective.lower()
    if obj.startswith("d"):
        dL_dx = -np.trace(SLinv @ dSL_x)
        dL_dy = -np.trace(SLinv @ dSL_y)
    else:
        dL_dx = -np.trace(SLinv @ dSL_x @ SLinv)
        dL_dy = -np.trace(SLinv @ dSL_y @ SLinv)

    if (not np.isfinite(dL_dx)) or (not np.isfinite(dL_dy)):
        return np.zeros(2, dtype=float)

    return np.array([float(dL_dx), float(dL_dy)], dtype=float)


class Adam2D:
    """Tiny Adam optimizer for sensor positions, stored as gradients -> direction only."""
    def __init__(self, lr=1.0, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = float(lr)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.eps = float(eps)
        self.t = 0
        self.m = None
        self.v = None

    def step(self, g: np.ndarray) -> np.ndarray:
        """
        g shape (S,2). Returns update direction (S,2).
        """
        if self.m is None:
            self.m = np.zeros_like(g)
            self.v = np.zeros_like(g)

        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)

        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)

        return -self.lr * mhat / (np.sqrt(vhat) + self.eps)
