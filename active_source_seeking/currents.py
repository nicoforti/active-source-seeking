from __future__ import annotations
import numpy as np

def bay_circulation_velocity(xy: np.ndarray) -> np.ndarray:
    """
    xy: (2, n) points
    returns v: (2, n) velocities
    A synthetic bay-like circulation using a streamfunction.
    """
    x = xy[0, :]
    y = xy[1, :]

    # Scale to roughly [0,1]
    x0 = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y0 = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # Streamfunction: combine a large gyre + a smaller eddy near bay/port region
    psi = (
        1.0 * np.sin(np.pi * x0) * np.sin(np.pi * y0)
        + 0.35 * np.exp(-((x0 - 0.75)**2 + (y0 - 0.45)**2) / 0.01)
    )

    # Numerical gradients in normalized coords
    # (for demos; for production you may want analytic derivatives)
    eps = 1e-3
    psi_x = (1.0 * np.sin(np.pi * (x0 + eps)) * np.sin(np.pi * y0)
             - 1.0 * np.sin(np.pi * (x0 - eps)) * np.sin(np.pi * y0)) / (2 * eps)
    psi_y = (1.0 * np.sin(np.pi * x0) * np.sin(np.pi * (y0 + eps))
             - 1.0 * np.sin(np.pi * x0) * np.sin(np.pi * (y0 - eps))) / (2 * eps)

    # v = [dpsi/dy, -dpsi/dx]
    v = np.vstack([psi_y, -psi_x])

    # Scale to m/s-like magnitude
    return 0.3 * v

