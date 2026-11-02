"""Demo meshes used to exercise the custom mesh input interface.

``square_mesh`` builds a structured mesh by hand (no gmsh): this is the
"pure" custom mesh path.  ``thermal1_mesh`` builds the exact geometry of
``examples/Thermal/Thermal1.py`` with EasyFEA/gmsh and converts it to
``MeshData`` so it can be fed back through the custom input interface.
"""

from __future__ import annotations

import env_compat  # noqa: F401  (must precede NumPy/EasyFEA)

import numpy as np

from EasyFEA import ElemType, Mesh
from EasyFEA.Geoms import Circle, Domain, Point

from custom_mesh import MeshData, as_elem_type, available_tags

__all__ = ["square_mesh", "u_shape_mesh", "u_shape_outline", "thermal1_mesh", "mesh_to_data"]


def square_mesh(
    n: int = 8,
    elem_type="TRI3",
    size: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    with_tags: bool = True,
) -> MeshData:
    """Structured ``n x n`` mesh of the square ``[0, size]^2``.

    Supported: TRI3, TRI6, QUAD4, QUAD8.  Node numbering is row-major, so the
    same topology can be written to CSV/JSON and reloaded.
    """

    elem_type = as_elem_type(elem_type)
    if elem_type not in (ElemType.TRI3, ElemType.TRI6, ElemType.QUAD4, ElemType.QUAD8):
        raise ValueError(f"square_mesh does not generate {elem_type}.")
    if n < 1:
        raise ValueError("n must be >= 1.")

    ox, oy = origin
    coords: list[list[float]] = []
    grid: dict[tuple[int, int], int] = {}

    for j in range(n + 1):
        for i in range(n + 1):
            grid[(i, j)] = len(coords)
            coords.append([ox + size * i / n, oy + size * j / n, 0.0])

    mid_cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in mid_cache:
            pa = np.asarray(coords[a])
            pb = np.asarray(coords[b])
            mid_cache[key] = len(coords)
            coords.append(((pa + pb) / 2.0).tolist())
        return mid_cache[key]

    elements: list[list[int]] = []

    for j in range(n):
        for i in range(n):
            c00 = grid[(i, j)]
            c10 = grid[(i + 1, j)]
            c11 = grid[(i + 1, j + 1)]
            c01 = grid[(i, j + 1)]

            if elem_type == ElemType.TRI3:
                elements.append([c00, c10, c11])
                elements.append([c00, c11, c01])
            elif elem_type == ElemType.TRI6:
                elements.append(
                    [
                        c00,
                        c10,
                        c11,
                        midpoint(c00, c10),
                        midpoint(c10, c11),
                        midpoint(c11, c00),
                    ]
                )
                elements.append(
                    [
                        c00,
                        c11,
                        c01,
                        midpoint(c00, c11),
                        midpoint(c11, c01),
                        midpoint(c01, c00),
                    ]
                )
            elif elem_type == ElemType.QUAD4:
                elements.append([c00, c10, c11, c01])
            elif elem_type == ElemType.QUAD8:
                elements.append(
                    [
                        c00,
                        c10,
                        c11,
                        c01,
                        midpoint(c00, c10),
                        midpoint(c10, c11),
                        midpoint(c11, c01),
                        midpoint(c01, c00),
                    ]
                )

    nodes = np.asarray(coords, dtype=float)
    conn = np.asarray(elements, dtype=np.int64)

    tags: dict[str, np.ndarray] = {}
    if with_tags:
        x, y = nodes[:, 0], nodes[:, 1]
        tol = 1e-12 * max(size, 1.0)
        tags = {
            "left": np.where(np.isclose(x, ox, atol=tol))[0],
            "right": np.where(np.isclose(x, ox + size, atol=tol))[0],
            "bottom": np.where(np.isclose(y, oy, atol=tol))[0],
            "top": np.where(np.isclose(y, oy + size, atol=tol))[0],
        }

    return MeshData(nodes=nodes, elements=conn, elem_type=elem_type, node_tags=tags)


def u_shape_outline(scale: float = 1.0) -> np.ndarray:
    """Outline of the U shape (base geometry 3 x 2, notch [1,2] x [1,2])."""

    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 2.0],
            [2.0, 2.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ]
    )
    return points * scale


def u_shape_mesh(
    n: int = 8,
    elem_type="QUAD4",
    scale: float = 1.0,
    with_tags: bool = True,
) -> MeshData:
    """Structured U-shaped mesh.

    The base geometry is ``[0, 3] x [0, 2]`` minus the notch
    ``[1, 2] x [1, 2]`` (area 5).  ``n`` is the number of cells per unit
    length, so the grid is ``3n x 2n`` with square cells and the notch
    boundaries coincide exactly with grid lines.

    Supported: TRI3, TRI6, QUAD4, QUAD8.
    """

    elem_type = as_elem_type(elem_type)
    if elem_type not in (
        ElemType.TRI3,
        ElemType.TRI6,
        ElemType.QUAD4,
        ElemType.QUAD8,
    ):
        raise ValueError(f"u_shape_mesh does not generate {elem_type}.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    nx, ny = 3 * n, 2 * n

    # Keep every cell of the 3x2 rectangle except the 1x1 notch.
    kept: list[tuple[int, int]] = []
    for j in range(ny):
        for i in range(nx):
            in_notch = (n <= i < 2 * n) and (n <= j < 2 * n)
            if not in_notch:
                kept.append((i, j))

    # Compact node numbering: only nodes used by kept cells (no orphans).
    used: set[tuple[int, int]] = set()
    for i, j in kept:
        used.update(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
    ordered = sorted(used, key=lambda node: (node[1], node[0]))
    node_id = {node: index for index, node in enumerate(ordered)}

    nodes: list[list[float]] = [
        [scale * i / n, scale * j / n, 0.0] for i, j in ordered
    ]

    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in midpoint_cache:
            pa = np.asarray(nodes[a])
            pb = np.asarray(nodes[b])
            midpoint_cache[key] = len(nodes)
            nodes.append(((pa + pb) / 2.0).tolist())
        return midpoint_cache[key]

    elements: list[list[int]] = []

    for i, j in kept:
        c00 = node_id[(i, j)]
        c10 = node_id[(i + 1, j)]
        c11 = node_id[(i + 1, j + 1)]
        c01 = node_id[(i, j + 1)]

        if elem_type == ElemType.QUAD4:
            elements.append([c00, c10, c11, c01])
        elif elem_type == ElemType.QUAD8:
            elements.append(
                [
                    c00,
                    c10,
                    c11,
                    c01,
                    midpoint(c00, c10),
                    midpoint(c10, c11),
                    midpoint(c11, c01),
                    midpoint(c01, c00),
                ]
            )
        elif elem_type == ElemType.TRI3:
            elements.append([c00, c10, c11])
            elements.append([c00, c11, c01])
        elif elem_type == ElemType.TRI6:
            elements.append(
                [
                    c00,
                    c10,
                    c11,
                    midpoint(c00, c10),
                    midpoint(c10, c11),
                    midpoint(c11, c00),
                ]
            )
            elements.append(
                [
                    c00,
                    c11,
                    c01,
                    midpoint(c00, c11),
                    midpoint(c11, c01),
                    midpoint(c01, c00),
                ]
            )

    node_array = np.asarray(nodes, dtype=float)
    conn = np.asarray(elements, dtype=np.int64)

    tags: dict[str, np.ndarray] = {}
    if with_tags:
        # Work in the normalized 3x2 frame so tolerances are scale independent.
        u = node_array[:, 0] / scale
        v = node_array[:, 1] / scale
        tol = 1e-12

        left = np.isclose(u, 0.0, atol=tol)
        right = np.isclose(u, 3.0, atol=tol)
        bottom = np.isclose(v, 0.0, atol=tol)
        top = np.isclose(v, 2.0, atol=tol)
        notch_left = np.isclose(u, 1.0, atol=tol) & (v >= 1.0 - tol)
        notch_right = np.isclose(u, 2.0, atol=tol) & (v >= 1.0 - tol)
        notch_bottom = (
            np.isclose(v, 1.0, atol=tol)
            & (u >= 1.0 - tol)
            & (u <= 2.0 + tol)
        )
        boundary = (
            left | right | bottom | top | notch_left | notch_right | notch_bottom
        )

        tags = {
            "left": np.where(left)[0],
            "right": np.where(right)[0],
            "bottom": np.where(bottom)[0],
            "top": np.where(top)[0],
            "notch_left": np.where(notch_left)[0],
            "notch_right": np.where(notch_right)[0],
            "notch_bottom": np.where(notch_bottom)[0],
            "boundary": np.where(boundary)[0],
        }

    return MeshData(
        nodes=node_array, elements=conn, elem_type=elem_type, node_tags=tags
    )


def mesh_to_data(mesh: Mesh, with_tags: bool = True) -> MeshData:
    """Convert an existing EasyFEA mesh (single 2D element group) to MeshData."""

    group_elem = mesh.groupElem
    data = MeshData(
        nodes=np.asarray(mesh.coord, dtype=float).copy(),
        elements=np.asarray(group_elem.connect, dtype=np.int64).copy(),
        elem_type=group_elem.elemType,
    )

    if with_tags:
        for tag in available_tags(mesh):
            nodes = mesh.Nodes_Tags(tag)
            if nodes.size > 0:
                data.node_tags[tag] = np.asarray(nodes, dtype=np.int64)

    return data


def thermal1_mesh(
    elem_type="TRI6", a: float = 1.0
) -> tuple[MeshData, Mesh]:
    """Reproduce the geometry of ``examples/Thermal/Thermal1.py``.

    Returns the ``MeshData`` of the 2D group together with the original gmsh
    mesh (used as the reference for the round-trip comparison).
    """

    elem_type = as_elem_type(elem_type)
    if elem_type not in (ElemType.TRI3, ElemType.TRI6):
        raise ValueError("Thermal1 uses a triangular mesh (TRI3 or TRI6).")

    domain = Domain(Point(), Point(a, a), a / 4)
    circle = Circle(Point(a / 2, a / 2), diam=a / 3, meshSize=a / 4)
    mesh = domain.Mesh_2D([circle], elem_type)

    # Keep only meaningful boundary tags; gmsh auto physical groups
    # (L0, P0, S0, ...) would otherwise clutter the interface.
    data = mesh_to_data(mesh, with_tags=False)

    # Recompute clean coordinate tags (gmsh physical groups are not needed).
    x = data.nodes[:, 0]
    tol = 1e-12 * max(a, 1.0)
    data.node_tags["left"] = np.where(np.isclose(x, 0.0, atol=tol))[0]
    data.node_tags["right"] = np.where(np.isclose(x, a, atol=tol))[0]

    return data, mesh
