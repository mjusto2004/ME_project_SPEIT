"""3D demo meshes for the custom mesh input interface.

- ``thermal2_mesh``: exact geometry of ``examples/Thermal/Thermal2.py``
  (square prism with a circular hole along z), generated with EasyFEA/gmsh
  and converted to ``MeshData``.
- ``hollow_box_mesh``: hand-built structured hexahedral mesh of a square
  prism with a square through-hole (no gmsh), useful as a pure custom test.
"""

from __future__ import annotations

import env_compat  # noqa: F401  (must precede NumPy/EasyFEA)

import numpy as np

from EasyFEA import ElemType, Mesh
from EasyFEA.Geoms import Circle, Domain, Point

from custom_mesh import (
    MeshData,
    SUPPORTED_3D_ELEM_TYPES,
    as_elem_type,
)
from demo_meshes import mesh_to_data

__all__ = ["thermal2_mesh", "hollow_box_mesh"]


def thermal2_mesh(
    elem_type="TETRA4", a: float = 1.0
) -> tuple[MeshData, Mesh]:
    """Reproduce the geometry of ``examples/Thermal/Thermal2.py``.

    Returns the 3D ``MeshData`` together with the original gmsh mesh (used as
    the reference for the round-trip comparison).
    """

    elem_type = as_elem_type(elem_type)
    if elem_type not in SUPPORTED_3D_ELEM_TYPES:
        raise ValueError("thermal2_mesh requires a 3D element type.")

    domain = Domain(Point(), Point(a, a), a / 4)
    circle = Circle(Point(a / 2, a / 2), diam=a / 3, meshSize=a / 4)
    mesh = domain.Mesh_Extrude([circle], [0, 0, -a], [4], elem_type)

    # Drop gmsh's automatic physical-group tags (L0, S0, ...).
    data = mesh_to_data(mesh, with_tags=False)

    # Collect every node that belongs to a lower-dimensional group: these are
    # exactly the boundary nodes of the 3D mesh (outer faces + hole surface).
    boundary_nodes: set[int] = set()
    for other_type, group_elem in mesh.dict_groupElem.items():
        if other_type != elem_type:
            boundary_nodes.update(np.asarray(group_elem.nodes, int).tolist())

    x = data.nodes[:, 0]
    z = data.nodes[:, 2]
    tol = 1e-12 * max(a, 1.0)
    data.node_tags["left"] = np.where(np.isclose(x, 0.0, atol=tol))[0]
    data.node_tags["right"] = np.where(np.isclose(x, a, atol=tol))[0]
    data.node_tags["zmin"] = np.where(np.isclose(z, z.min(), atol=tol))[0]
    data.node_tags["zmax"] = np.where(np.isclose(z, z.max(), atol=tol))[0]
    data.node_tags["boundary"] = np.asarray(sorted(boundary_nodes), dtype=np.int64)

    return data, mesh


def hollow_box_mesh(
    n: int = 6,
    nz: int = 2,
    scale: float = 1.0,
    with_tags: bool = True,
) -> MeshData:
    """Structured HEXA8 mesh of a square prism with a square through-hole.

    Base geometry: outer ``[0,3] x [0,3] x [0,1]`` minus the through-hole
    ``[1,2] x [1,2] x [0,1]``.  Volume = ``8 * scale**3``.
    """

    if n < 1:
        raise ValueError("n must be >= 1.")
    if nz < 1:
        raise ValueError("nz must be >= 1.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    nx = ny = 3 * n

    # Keep every (i,j) cell of the 3x3 cross-section except the 1x1 hole.
    kept = [
        (i, j)
        for j in range(ny)
        for i in range(nx)
        if not (n <= i < 2 * n and n <= j < 2 * n)
    ]

    used: set[tuple[int, int, int]] = set()
    for k in range(nz + 1):
        for i, j in kept:
            used.update(
                ((i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k))
            )

    ordered = sorted(used, key=lambda node: (node[2], node[1], node[0]))
    node_id = {node: index for index, node in enumerate(ordered)}
    nodes = np.asarray(
        [
            [scale * i / n, scale * j / n, scale * k / nz]
            for i, j, k in ordered
        ],
        dtype=float,
    )

    elements = []
    for k in range(nz):
        for i, j in kept:
            b00 = node_id[(i, j, k)]
            b10 = node_id[(i + 1, j, k)]
            b11 = node_id[(i + 1, j + 1, k)]
            b01 = node_id[(i, j + 1, k)]
            t00 = node_id[(i, j, k + 1)]
            t10 = node_id[(i + 1, j, k + 1)]
            t11 = node_id[(i + 1, j + 1, k + 1)]
            t01 = node_id[(i, j + 1, k + 1)]
            elements.append([b00, b10, b11, b01, t00, t10, t11, t01])

    conn = np.asarray(elements, dtype=np.int64)

    tags: dict[str, np.ndarray] = {}
    if with_tags:
        u = nodes[:, 0] / scale
        v = nodes[:, 1] / scale
        w = nodes[:, 2] / scale
        tol = 1e-12

        left = np.isclose(u, 0.0, atol=tol)
        right = np.isclose(u, 3.0, atol=tol)
        front = np.isclose(v, 0.0, atol=tol)
        back = np.isclose(v, 3.0, atol=tol)
        bottom = np.isclose(w, 0.0, atol=tol)
        top = np.isclose(w, 1.0, atol=tol)

        side_x = (np.isclose(u, 1.0, atol=tol) | np.isclose(u, 2.0, atol=tol)) & (
            (v >= 1.0 - tol) & (v <= 2.0 + tol)
        )
        side_y = (np.isclose(v, 1.0, atol=tol) | np.isclose(v, 2.0, atol=tol)) & (
            (u >= 1.0 - tol) & (u <= 2.0 + tol)
        )
        inner = side_x | side_y
        boundary = left | right | front | back | bottom | top | inner

        tags = {
            "left": np.where(left)[0],
            "right": np.where(right)[0],
            "front": np.where(front)[0],
            "back": np.where(back)[0],
            "bottom": np.where(bottom)[0],
            "top": np.where(top)[0],
            "inner": np.where(inner)[0],
            "boundary": np.where(boundary)[0],
        }

    return MeshData(
        nodes=nodes, elements=conn, elem_type=ElemType.HEXA8, node_tags=tags
    )
