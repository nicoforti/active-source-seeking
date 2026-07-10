# src/linalg_utils.py
from __future__ import annotations
import numpy as np
import scipy.linalg as la

def symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def cho_factor_spd(A: np.ndarray, base_jitter: float = 1e-12, max_tries: int = 10):
    """
    Robust Cholesky: symmetrize and add increasing diagonal jitter until SPD.
    Returns (cF, A_spd, jitter_used).
    """
    A = symmetrize(A)
    # scale jitter to matrix magnitude
    scale = max(1.0, float(np.max(np.abs(np.diag(A)))) )
    jitter = base_jitter * scale

    I = np.eye(A.shape[0], dtype=A.dtype)
    for _ in range(max_tries):
        try:
            cF = la.cho_factor(A + jitter * I, lower=True, check_finite=False)
            return cF, A + jitter * I, jitter
        except la.LinAlgError:
            jitter *= 10.0

    # last attempt: stronger regularization
    jitter = max(jitter, 1e-6 * scale)
    cF = la.cho_factor(A + jitter * I, lower=True, check_finite=False)
    return cF, A + jitter * I, jitter
