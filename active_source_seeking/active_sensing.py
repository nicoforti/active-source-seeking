# src/active_sensing.py
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from active_source_seeking.fem_sampling_p1 import fe_p1_triplet


def build_measurement_information_free_sparse(mesh, stepper, sensors_xy: np.ndarray, y: np.ndarray, r: np.ndarray, finder=None):
    """
    Build Lambda and lambda in information form using only free dofs.

    Lambda = sum_i (1/r_i) * c_i^T c_i
    lambda = sum_i (1/r_i) * c_i^T y_i

    c_i is the P1 sampling row phi(s_i)^T (3 nonzeros on containing triangle).
    We build Lambda as sparse (CSR), with 9 contributions per sensor,
    mapped to free dofs (Dirichlet dofs are dropped).
    """
    if finder is None:
        finder = mesh.element_finder()

    nfree = stepper.free.shape[0]
    g2f = -np.ones(stepper.N, dtype=int)
    g2f[stepper.free] = np.arange(nfree, dtype=int)

    lam = np.zeros(nfree, dtype=float)

    rows_all = []
    cols_all = []
    data_all = []

    S = sensors_xy.shape[0]
    for i in range(S):
        try:
            tri_nodes, Nhat, _ = fe_p1_triplet(mesh, sensors_xy[i], finder=finder)
        except ValueError:
            continue
        wi = 1.0 / float(r[i])
        yi = float(y[i])

        fidx = g2f[tri_nodes]  # (3,) may contain -1

        for a in range(3):
            ia = int(fidx[a])
            if ia < 0:
                continue
            lam[ia] += wi * float(Nhat[a]) * yi

        for a in range(3):
            ia = int(fidx[a])
            if ia < 0:
                continue
            for b in range(3):
                ib = int(fidx[b])
                if ib < 0:
                    continue
                rows_all.append(ia)
                cols_all.append(ib)
                data_all.append(wi * float(Nhat[a]) * float(Nhat[b]))

    Lambda = sp.coo_matrix((data_all, (rows_all, cols_all)), shape=(nfree, nfree)).tocsr()
    return Lambda, lam
