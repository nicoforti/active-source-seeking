from __future__ import annotations
import numpy as np
from skfem import MeshTri
from .fem_sampling_p1 import fe_row_and_grad_p1

def source_injection_vector(mesh: MeshTri, rho: np.ndarray, finder=None):
    """
    Returns:
      b   : (N,) source injection vector (P1 row)
      dbdx: (N,) derivative wrt rho_x
      dbdy: (N,) derivative wrt rho_y
    """
    c, cx, cy, _ = fe_row_and_grad_p1(mesh, rho, finder=finder)
    return c.copy(), cx.copy(), cy.copy()


