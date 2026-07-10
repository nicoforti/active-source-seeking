import numpy as np

def weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w[:, None] * x).sum(axis=0)

def weighted_cov_2d(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = weighted_mean(x, w)
    xc = x - mu[None, :]
    C = (w[:, None, None] * (xc[:, :, None] @ xc[:, None, :])).sum(axis=0)
    return C

def uncertainty_radius_2sigma(z: np.ndarray, w: np.ndarray) -> float:
    """
    2-sigma radius in meters using max eigenvalue of 2D covariance.
    r = 2 * sqrt(lambda_max(Cov(rho))).
    """
    C = weighted_cov_2d(z[:, 0:2], w)
    eigs = np.linalg.eigvalsh(C)
    lam_max = float(np.max(eigs))
    return 2.0 * np.sqrt(max(lam_max, 0.0))
