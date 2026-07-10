from __future__ import annotations

# src/fem_sampling_p1.py

import numpy as np
from skfem import MeshTri

def _nearest_vertex(mesh, s):
    # mesh.p shape: (2, nverts)
    p = mesh.p.T  # (nverts, 2)
    d2 = np.sum((p - s[None, :])**2, axis=1)
    j = np.argmin(d2)
    return p[j]

def _find_element(mesh, s: np.ndarray, finder=None) -> int:
    if finder is None:
        finder = mesh.element_finder()
    try:
        e = finder(np.array([s[0]]), np.array([s[1]]))
        e = int(np.atleast_1d(e)[0])
    except ValueError as ex:
        raise ValueError("Point is outside mesh") from ex
    if e < 0:
        raise ValueError("Point is outside mesh")
    return e


def fe_p1_triplet(mesh: MeshTri, s: np.ndarray, finder=None):
    """
    Returns:
      tri_nodes : (3,) global node indices
      Nhat      : (3,) barycentric values at s
      e         : element index
    """
    s = np.asarray(s, dtype=float).reshape(2,)
    e = _find_element(mesh, s, finder=finder)

    tri_nodes = mesh.t[:, e].astype(int)
    X = mesh.p[:, tri_nodes]  # (2,3)

    x1 = X[:, 0]
    T = np.column_stack([X[:, 1] - x1, X[:, 2] - x1])  # (2,2)
    rhs = s - x1
    l2, l3 = np.linalg.solve(T, rhs)
    l1 = 1.0 - l2 - l3
    Nhat = np.array([l1, l2, l3], dtype=float)

    return tri_nodes, Nhat, e


def fe_row_and_grad_p1(mesh, s: np.ndarray, finder=None, eps_det: float = 1e-14):
    """
    Returns:
      c  : length-N row with phi(s)^T (3 nonzeros)
      cx : length-N row with d(phi^T)/dx at s (3 nonzeros; constant on triangle)
      cy : length-N row with d(phi^T)/dy at s
      e  : element index
    """
    s = np.asarray(s, dtype=float).reshape(2,)
    e = _find_element(mesh, s, finder=finder)

    tri_nodes = mesh.t[:, e].astype(int)
    X = mesh.p[:, tri_nodes]  # (2,3)

    x1 = X[:, 0]
    J = np.column_stack([X[:, 1] - x1, X[:, 2] - x1])  # (2,2)

    # degeneracy check (coarse meshes can have skinny triangles)
    detJ = float(np.linalg.det(J))
    if abs(detJ) < eps_det:
        # return "no contribution" rather than crashing
        N = mesh.p.shape[1]
        return np.zeros(N), np.zeros(N), np.zeros(N), e

    rhs = s - x1
    try:
        l2, l3 = np.linalg.solve(J, rhs)
    except np.linalg.LinAlgError:
        N = mesh.p.shape[1]
        return np.zeros(N), np.zeros(N), np.zeros(N), e

    l1 = 1.0 - l2 - l3
    Nhat = np.array([l1, l2, l3], dtype=float)

    # gradients: grad_phys = grad_ref @ inv(J).T  (but do without explicit inverse)
    grad_ref = np.array(
        [[-1.0, -1.0],
         [ 1.0,  0.0],
         [ 0.0,  1.0]],
        dtype=float
    )  # (3,2)

    try:
        invJT = np.linalg.solve(J.T, np.eye(2))  # = inv(J).T
    except np.linalg.LinAlgError:
        N = mesh.p.shape[1]
        return np.zeros(N), np.zeros(N), np.zeros(N), e

    grad_phys = grad_ref @ invJT  # (3,2)

    N = mesh.p.shape[1]
    c = np.zeros(N, dtype=float)
    cx = np.zeros(N, dtype=float)
    cy = np.zeros(N, dtype=float)

    c[tri_nodes] = Nhat
    cx[tri_nodes] = grad_phys[:, 0]
    cy[tri_nodes] = grad_phys[:, 1]

    return c, cx, cy, e


