# src/fem_advection_diffusion.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from skfem import BilinearForm, MeshTri, Basis, asm, condense
from skfem.helpers import dot, grad
from skfem.element import ElementTriP1

from .currents import bay_circulation_velocity

# -------------------------
# Bilinear forms (P1)
# -------------------------

@BilinearForm
def mass_form(u, v, w):
    return u * v


@BilinearForm
def diff_form(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def adv_form(u, v, w):
    beta = w["beta"]  # shape (2, nqp_total)
    return (beta[0] * grad(u)[0] + beta[1] * grad(u)[1]) * v


# -------------------------
# Model container
# -------------------------

@dataclass
class AdvectionDiffusionModel:
    mesh: MeshTri
    basis: Basis
    M: sp.csr_matrix
    K: sp.csr_matrix
    Aadv: sp.csr_matrix
    boundaries: dict


@dataclass
class ImplicitEulerStepper:
    # full size N, but we work on free dofs
    N: int
    D: np.ndarray              # Dirichlet dofs (1d int array)
    free: np.ndarray           # free dofs (1d int array)

    # condensed matrices (nfree x nfree)
    M_free: sp.csr_matrix
    LHS_free: sp.csr_matrix
    lu: splu                   # LU factorization of LHS_free

    # deterministic map x_{k+1} = A x_k (+ forcing)
    A_free: np.ndarray         # dense (nfree x nfree) -- built once, n~1e3 ok

    # process noise model on free dofs
    Q_free: sp.csr_matrix      # (nfree x nfree) SPD
    Qinv_free: sp.csr_matrix   # inverse (sparse diagonal in our choice)

    def solve_free(self, rhs_free: np.ndarray) -> np.ndarray:
        return self.lu.solve(rhs_free)

    def A_apply_free(self, x_free: np.ndarray) -> np.ndarray:
        return self.solve_free(self.M_free @ x_free)

    def expand_full(self, x_free: np.ndarray) -> np.ndarray:
        x = np.zeros(self.N)
        x[self.free] = x_free
        x[self.D] = 0.0
        return x

    def restrict_full(self, x_full: np.ndarray) -> np.ndarray:
        return x_full[self.free]


# -------------------------
# Assembly
# -------------------------

def assemble_advection_diffusion(
    mesh: MeshTri,
    boundaries: dict,
    diffusivity: float = 5.0,
    beta_fun=bay_circulation_velocity,
) -> AdvectionDiffusionModel:
    basis = Basis(mesh, ElementTriP1())

    xqp = basis.global_coordinates()    # (2, nqp_total)
    beta = beta_fun(xqp)                # (2, nqp_total)

    M = asm(mass_form, basis).tocsr()
    K = (diffusivity * asm(diff_form, basis)).tocsr()
    Aadv = asm(adv_form, basis, beta=beta).tocsr()

    return AdvectionDiffusionModel(mesh=mesh, basis=basis, M=M, K=K, Aadv=Aadv, boundaries=boundaries)


# -------------------------
# Time discretization + Q model + A matrix on free dofs
# -------------------------

def time_discretize_implicit_euler(
    model: AdvectionDiffusionModel,
    dt: float,
    dirichlet_tag: str = "open_sea",
    sigma_w: float = 1e-2,
):
    """
    Implicit Euler:
      (M + dt*(K + Aadv)) x_{k+1} = M x_k + dt f_k

    We impose Dirichlet=0 on 'dirichlet_tag', then work on free dofs only.

    Process noise suggestion (SPD, easy inverse):
      Q_free = sigma_w^2 * diag(M_lumped_free)^{-1}
    i.e., spatially white noise in L2 -> covariance proportional to M^{-1}.
    Then Qinv_free is diagonal: (1/sigma_w^2)*diag(M_lumped_free).
    """

    ALIASES = {
    "open_sea": "open_sea_boundary",
    "coast": "coast_boundary",
    "port": "port_boundary",
    }

    if dirichlet_tag in ALIASES:
        dirichlet_tag = ALIASES[dirichlet_tag]

    basis = model.basis
    N = model.M.shape[0]

    if dirichlet_tag not in model.boundaries:
        raise ValueError(f"Boundary tag '{dirichlet_tag}' not found. Available: {list(model.boundaries.keys())}")

    facets = model.boundaries[dirichlet_tag]
    D = basis.get_dofs(facets=facets).all().astype(int)
    all_idx = np.arange(N, dtype=int)
    free = np.setdiff1d(all_idx, D)

    M = model.M
    G = (model.K + model.Aadv).tocsr()
    LHS = (M + dt * G).tocsr()

    # Condense matrices to free dofs (Dirichlet=0)
    # condense(A, D=...) returns (A_free, b_free, x_bc)
    b0 = np.zeros(N, dtype=float)
    x0 = np.zeros(N, dtype=float)

    out = condense(LHS, b=b0, D=D, x=x0)
    LHS_free = out[0]
    # out[1] = b_free (non ci serve qui)
    # out[2] = x_bc  (non ci serve qui)
    # out[3] = I/free (se presente)

    outM = condense(M, b=b0, D=D, x=x0)
    M_free = outM[0]

    LHS_free = LHS_free.tocsr()
    M_free = M_free.tocsr()
    lu = splu(LHS_free.tocsc())

    # Build A_free = LHS_free^{-1} M_free (dense OK at n~1e3)
    # Solve for many RHS at once: LU.solve expects dense (nfree, nrhs)
    M_free_dense = M_free.toarray()
    A_free = lu.solve(M_free_dense)  # returns dense ndarray (nfree, nfree)

    # Lumped mass diag on free dofs: diag(M_free * 1)
    ones = np.ones(M_free.shape[0])
    M_lumped = np.asarray(M_free @ ones).ravel()
    # Avoid zeros (shouldn't happen if mesh ok)
    M_lumped = np.maximum(M_lumped, 1e-12)

    # Q_free = sigma_w^2 * diag(1 / M_lumped)
    Q_diag = (sigma_w**2) * (1.0 / M_lumped)
    Qinv_diag = (1.0 / (sigma_w**2)) * M_lumped

    Q_free = sp.diags(Q_diag, format="csr")
    Qinv_free = sp.diags(Qinv_diag, format="csr")

    stepper = ImplicitEulerStepper(
        N=N, D=D, free=free,
        M_free=M_free, LHS_free=LHS_free, lu=lu,
        A_free=A_free,
        Q_free=Q_free, Qinv_free=Qinv_free,
    )
    return stepper
