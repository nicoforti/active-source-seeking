from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import meshio
from skfem import MeshTri

@dataclass
class LoadedMesh:
    mesh: MeshTri
    boundaries: Dict[str, np.ndarray]  # name -> facet indices
    cell_data: dict

def load_gmsh_tri(path: str):
    msh = meshio.read(path)

    # --- triangles (domain) ---
    tri = msh.cells_dict.get("triangle", None)
    if tri is None:
        raise ValueError("No triangle cells found in msh.")
    p = msh.points[:, :2].T  # (2, npts)
    t = tri.T.astype(np.int32)  # (3, ntri)
    mesh = MeshTri(p, t)

    # --- build id->name maps from field_data ---
    # field_data: name -> (id, dim)
    id_to_name_1d = {}
    if hasattr(msh, "field_data") and msh.field_data:
        for name, (pid, dim) in msh.field_data.items():
            if int(dim) == 1:
                id_to_name_1d[int(pid)] = name

    # --- line physical ids ---
    boundaries = {}
    line = msh.cells_dict.get("line", None)
    if line is not None:
        # physical id per ogni line element
        phys = msh.cell_data_dict["gmsh:physical"]["line"]
        phys = np.asarray(phys).astype(int)

        # map each physical id to facet indices in skfem
        # skfem facets are edges: mesh.facets is (2, nfacets) of node indices
        # We match line segments by node pairs (unordered)
        facets = mesh.facets.T  # (nfacets, 2)
        facet_map = {tuple(sorted(f)): i for i, f in enumerate(facets)}

        for k, seg in enumerate(line):
            n0, n1 = int(seg[0]), int(seg[1])
            key = tuple(sorted((n0, n1)))
            if key not in facet_map:
                continue
            fid = facet_map[key]
            pid = int(phys[k])
            name = id_to_name_1d.get(pid, f"phys_{pid}")
            boundaries.setdefault(name, []).append(fid)

        # convert lists to numpy arrays
        for name in list(boundaries.keys()):
            boundaries[name] = np.asarray(boundaries[name], dtype=int)

    else:
        # no line elements -> no boundary tags
        boundaries = {}

    class Loaded:
        def __init__(self, mesh, boundaries):
            self.mesh = mesh
            self.boundaries = boundaries

    return Loaded(mesh, boundaries)
