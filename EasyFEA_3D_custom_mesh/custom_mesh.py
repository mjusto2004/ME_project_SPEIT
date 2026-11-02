"""Custom 2D/3D mesh input interface for EasyFEA.

The standard EasyFEA examples build their mesh from geometric objects
(``Domain``, ``Circle`` ...) through gmsh.  This module replaces that step:
it takes a user supplied nodal cloud + element connectivity (in memory, JSON,
NPZ or CSV), validates it, and builds a regular ``EasyFEA.Mesh`` that can be
passed to ``Simulations.Thermal`` exactly like a gmsh generated mesh.

Supported 2D element types: TRI3, TRI6, QUAD4, QUAD8.
Supported 3D element types: TETRA4, TETRA10, HEXA8, HEXA20, HEXA27,
PRISM6, PRISM15, PRISM18.

Node ordering follows the gmsh/EasyFEA convention:

    TRI3    : v0, v1, v2
    TRI6    : v0, v1, v2, m01, m12, m20
    QUAD4   : v0, v1, v2, v3
    QUAD8   : v0, v1, v2, v3, m01, m12, m23, m30
    TETRA4  : v0, v1, v2, v3
    TETRA10 : v0..v3, then edge nodes (gmsh order)
    HEXA8   : v0..v3 (bottom), v4..v7 (top)
    HEXA20  : HEXA8 + 12 edge nodes (gmsh order)
    HEXA27  : HEXA20 + face/volume nodes (gmsh order)
    PRISM6  : v0..v2 (bottom triangle), v3..v5 (top triangle)
    PRISM15 : PRISM6 + 9 edge nodes (gmsh order)
    PRISM18 : PRISM15 + 3 face nodes (gmsh order)

All coordinates are 3D inside EasyFEA; a 2D input is padded with z = 0.
"""

from __future__ import annotations

import env_compat  # noqa: F401  (must precede NumPy/EasyFEA)

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from EasyFEA import ElemType, MatrixType, Mesh
from EasyFEA.FEM import GroupElemFactory

__all__ = [
    "MeshInputError",
    "MeshData",
    "SUPPORTED_2D_ELEM_TYPES",
    "SUPPORTED_3D_ELEM_TYPES",
    "SUPPORTED_ELEM_TYPES",
    "as_elem_type",
    "build_mesh",
    "load_mesh",
    "load_mesh_data",
    "load_nodes_elements",
    "save_mesh_data",
    "save_mesh_csv",
    "mesh_info",
    "available_tags",
]

SUPPORTED_2D_ELEM_TYPES = (
    ElemType.TRI3,
    ElemType.TRI6,
    ElemType.QUAD4,
    ElemType.QUAD8,
)

SUPPORTED_3D_ELEM_TYPES = (
    ElemType.TETRA4,
    ElemType.TETRA10,
    ElemType.HEXA8,
    ElemType.HEXA20,
    ElemType.HEXA27,
    ElemType.PRISM6,
    ElemType.PRISM15,
    ElemType.PRISM18,
)

SUPPORTED_ELEM_TYPES = SUPPORTED_2D_ELEM_TYPES + SUPPORTED_3D_ELEM_TYPES

_NPE = {
    ElemType.TRI3: 3,
    ElemType.TRI6: 6,
    ElemType.QUAD4: 4,
    ElemType.QUAD8: 8,
    ElemType.TETRA4: 4,
    ElemType.TETRA10: 10,
    ElemType.HEXA8: 8,
    ElemType.HEXA20: 20,
    ElemType.HEXA27: 27,
    ElemType.PRISM6: 6,
    ElemType.PRISM15: 15,
    ElemType.PRISM18: 18,
}

# number of corner (vertex) nodes used to compute the signed area/volume
_NCORNER = {
    ElemType.TRI3: 3,
    ElemType.TRI6: 3,
    ElemType.QUAD4: 4,
    ElemType.QUAD8: 4,
    ElemType.TETRA4: 4,
    ElemType.TETRA10: 4,
    ElemType.HEXA8: 8,
    ElemType.HEXA20: 8,
    ElemType.HEXA27: 8,
    ElemType.PRISM6: 6,
    ElemType.PRISM15: 6,
    ElemType.PRISM18: 6,
}

# local permutation that flips a clockwise/negative element.
# Only linear types are listed; higher-order elements must be given with the
# correct (positive) orientation.
_REVERSE = {
    ElemType.TRI3: [0, 2, 1],
    ElemType.TRI6: [0, 2, 1, 5, 4, 3],
    ElemType.QUAD4: [0, 3, 2, 1],
    ElemType.QUAD8: [0, 3, 2, 1, 7, 6, 5, 4],
    ElemType.TETRA4: [0, 2, 1, 3],
    ElemType.HEXA8: [0, 3, 2, 1, 4, 7, 6, 5],
    ElemType.PRISM6: [0, 2, 1, 3, 5, 4],
}


class MeshInputError(ValueError):
    """Raised when user supplied mesh data is not usable by EasyFEA."""


def as_elem_type(value) -> ElemType:
    """Convert a string (``"TRI3"``) or ``ElemType`` to ``ElemType``."""

    if isinstance(value, ElemType):
        elem_type = value
    elif isinstance(value, str):
        try:
            elem_type = ElemType[value.strip().upper()]
        except KeyError as exc:
            raise MeshInputError(f"Unknown element type {value!r}.") from exc
    else:
        raise MeshInputError(f"Invalid element type {value!r}.")

    if elem_type not in SUPPORTED_ELEM_TYPES:
        raise MeshInputError(
            f"{elem_type} is not a supported element type. "
            f"2D: {[str(e) for e in SUPPORTED_2D_ELEM_TYPES]}; "
            f"3D: {[str(e) for e in SUPPORTED_3D_ELEM_TYPES]}."
        )
    return elem_type


@dataclass
class MeshData:
    """A user defined 2D/3D mesh (nodes, connectivity and optional node tags)."""

    nodes: np.ndarray
    elements: np.ndarray
    elem_type: ElemType
    node_tags: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self):
        self.elem_type = as_elem_type(self.elem_type)
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.elements = np.asarray(self.elements)
        if not isinstance(self.node_tags, dict):
            raise MeshInputError("node_tags must be a dict {tag: node indices}.")
        self.node_tags = {
            str(tag): np.asarray(nodes, dtype=np.int64).ravel()
            for tag, nodes in self.node_tags.items()
        }

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0] if self.elements.ndim == 2 else 0)

    def __repr__(self) -> str:
        return (
            f"MeshData({self.elem_type}, Nn={self.n_nodes}, "
            f"Ne={self.n_elements}, tags={sorted(self.node_tags)})"
        )


# ---------------------------------------------------------------------------
# validation / mesh construction
# ---------------------------------------------------------------------------


def _prepare_nodes(nodes: np.ndarray, elem_type: ElemType) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)

    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
        raise MeshInputError(
            f"nodes must be a (Nn, 2) or (Nn, 3) array, got {nodes.shape}."
        )
    if nodes.shape[0] == 0:
        raise MeshInputError("nodes is empty.")
    if not np.all(np.isfinite(nodes)):
        bad = np.where(~np.all(np.isfinite(nodes), axis=1))[0][:10]
        raise MeshInputError(f"nodes contain NaN/Inf (rows {bad.tolist()}).")

    if elem_type in SUPPORTED_3D_ELEM_TYPES:
        if nodes.shape[1] != 3:
            raise MeshInputError(
                f"{elem_type} is a 3D element type; nodes must be (Nn, 3), "
                f"got {nodes.shape}."
            )
        return nodes.copy()

    # 2D element type: pad a 2D input, or require a planar 3D input.
    if nodes.shape[1] == 2:
        return np.column_stack([nodes, np.zeros(nodes.shape[0])])

    z = nodes[:, 2]
    span = float(np.ptp(nodes[:, :2])) if nodes.shape[0] > 1 else 1.0
    if float(np.ptp(z)) > 1e-9 * max(1.0, span):
        raise MeshInputError(
            "A 2D mesh must be planar: the z coordinate is not constant."
        )
    out = nodes.copy()
    out[:, 2] = 0.0
    return out


def _prepare_elements(
    elements: np.ndarray,
    elem_type: ElemType,
    n_nodes: int,
    index_base: int,
) -> np.ndarray:
    arr = np.asarray(elements)
    n_pe = _NPE[elem_type]

    if arr.ndim != 2 or arr.shape[1] != n_pe:
        raise MeshInputError(
            f"{elem_type} expects a (Ne, {n_pe}) connectivity array, got {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise MeshInputError("elements is empty.")

    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)) or not np.allclose(arr, np.round(arr)):
            raise MeshInputError("element connectivity must contain integers.")
    arr = arr.astype(np.int64)

    if index_base not in (0, 1):
        raise MeshInputError("index_base must be 0 or 1.")
    if index_base == 1:
        arr = arr - 1

    if arr.min() < 0 or arr.max() >= n_nodes:
        raise MeshInputError(
            f"element node indices must be in [0, {n_nodes - 1}], "
            f"got [{arr.min()}, {arr.max()}] (index_base={index_base})."
        )

    duplicate = (np.diff(np.sort(arr, axis=1), axis=1) == 0).any(axis=1)
    if duplicate.any():
        rows = np.where(duplicate)[0][:10].tolist()
        raise MeshInputError(
            f"elements contain repeated node indices (elements {rows})."
        )

    return arr


def _signed_areas(nodes: np.ndarray, elements: np.ndarray, elem_type: ElemType):
    n_corner = _NCORNER[elem_type]
    p = nodes[elements[:, :n_corner], :2]
    x, y = p[..., 0], p[..., 1]
    return 0.5 * np.sum(
        x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1
    )


def build_mesh(
    data: MeshData,
    *,
    fix_orientation: bool = True,
    check_quality: bool = True,
    verbose: bool = False,
) -> Mesh:
    """Validate ``data`` and return an ``EasyFEA.Mesh``.

    Negative (clockwise) elements are reversed when ``fix_orientation`` is True;
    mixed orientations and degenerate elements always raise ``MeshInputError``.
    """

    dim = 3 if data.elem_type in SUPPORTED_3D_ELEM_TYPES else 2
    nodes = _prepare_nodes(data.nodes, data.elem_type)
    elements = _prepare_elements(data.elements, data.elem_type, len(nodes), 0)

    if dim == 2:
        areas = _signed_areas(nodes, elements, data.elem_type)

        span = float(np.ptp(nodes[:, :2]))
        area_tol = 1e-12 * max(1.0, span) ** 2
        degenerate = np.abs(areas) <= area_tol
        if degenerate.any():
            rows = np.where(degenerate)[0][:10].tolist()
            raise MeshInputError(f"degenerate (zero area) elements: {rows}.")

        negative = areas < 0
        if negative.any():
            if not negative.all():
                rows = np.where(negative)[0][:10].tolist()
                raise MeshInputError(
                    "elements have mixed orientation "
                    "(clockwise and counter-clockwise); "
                    f"first clockwise elements: {rows}."
                )
            if not fix_orientation:
                raise MeshInputError(
                    "all elements are clockwise; set fix_orientation=True "
                    "or reverse them."
                )
            if data.elem_type not in _REVERSE:
                raise MeshInputError(
                    f"automatic orientation fix is not implemented for "
                    f"{data.elem_type}; please provide counter-clockwise elements."
                )
            elements = elements[:, _REVERSE[data.elem_type]]
            warnings.warn(
                f"Reversed the connectivity of {len(elements)} clockwise "
                f"{data.elem_type} elements.",
                stacklevel=2,
            )

        group_elem = GroupElemFactory.Create(data.elem_type, elements, nodes)

    else:
        group_elem = GroupElemFactory.Create(data.elem_type, elements, nodes)
        jacobian = np.asarray(
            group_elem.Get_jacobian_e_pg(
                MatrixType.mass, absoluteValues=False
            )
        )
        jacobian_scale = max(float(np.max(np.abs(jacobian))), 1.0)
        jacobian_tol = 1e-12 * jacobian_scale

        if np.any(np.min(np.abs(jacobian), axis=1) <= jacobian_tol):
            rows = np.where(
                np.min(np.abs(jacobian), axis=1) <= jacobian_tol
            )[0][:10].tolist()
            raise MeshInputError(f"degenerate (zero volume) elements: {rows}.")

        negative = (jacobian < 0).any(axis=1)
        positive = (jacobian > 0).any(axis=1)
        if negative.any() and positive.any():
            rows = np.where(negative)[0][:10].tolist()
            raise MeshInputError(
                "elements have mixed Jacobian orientation (negative/positive); "
                f"first negative elements: {rows}."
            )
        if negative.any():
            if not fix_orientation:
                raise MeshInputError(
                    "all elements have a negative Jacobian; set "
                    "fix_orientation=True or reverse them."
                )
            if data.elem_type not in _REVERSE:
                raise MeshInputError(
                    f"automatic orientation fix is not implemented for "
                    f"{data.elem_type}; please provide positive-volume elements."
                )
            elements = elements[:, _REVERSE[data.elem_type]]
            group_elem = GroupElemFactory.Create(
                data.elem_type, elements, nodes
            )
            warnings.warn(
                f"Reversed the connectivity of {len(elements)} "
                f"{data.elem_type} elements with negative volume.",
                stacklevel=2,
            )

    mesh = Mesh({data.elem_type: group_elem}, verbosity=False)

    for tag, tag_nodes in data.node_tags.items():
        tag_nodes = np.asarray(tag_nodes, dtype=np.int64).ravel()
        if tag_nodes.size == 0:
            continue
        if tag_nodes.min() < 0 or tag_nodes.max() >= len(nodes):
            raise MeshInputError(
                f"tag {tag!r}: node indices must be in [0, {len(nodes) - 1}]."
            )
        mesh.Set_Tag(tag_nodes, tag)

    if len(mesh.orphanNodes) > 0:
        warnings.warn(
            f"{len(mesh.orphanNodes)} orphan node(s) are not used by any element.",
            stacklevel=2,
        )

    if check_quality:
        quality = mesh.Get_Quality("aspect")
        if quality is not None and np.nanmin(quality) <= 1e-12:
            warnings.warn(
                "At least one element has a near-zero aspect quality; "
                "check the connectivity.",
                stacklevel=2,
            )

    if verbose:
        info = mesh_info(mesh)
        quality = info["quality_min"]
        quality_text = f"{quality:.3g}" if quality is not None else "n/a"
        measure_key = "volume" if dim == 3 else "area"
        print(
            f"[custom mesh] {info['elem_type']}: Nn={info['Nn']}, "
            f"Ne={info['Ne']}, {measure_key}={info[measure_key]:.6g}, "
            f"min quality={quality_text}, tags={info['tags']}"
        )

    return mesh


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def _read_table(path: str | Path) -> np.ndarray:
    """Read a numeric table from CSV (comma) or plain text (whitespace).

    A single header line is allowed and detected automatically.
    """

    path = Path(path)
    if not path.is_file():
        raise MeshInputError(f"File not found: {path}")

    for delimiter in (",", None):
        for skip_rows in (0, 1):
            try:
                table = np.loadtxt(
                    path,
                    delimiter=delimiter,
                    comments="#",
                    skiprows=skip_rows,
                    ndmin=2,
                )
            except (ValueError, IndexError, StopIteration):
                continue
            if table.size > 0 and np.all(np.isfinite(table)):
                return np.asarray(table, dtype=float)

    raise MeshInputError(
        f"Could not read a numeric table from {path}. "
        "Expected rows of numbers separated by commas or whitespace, "
        "with an optional single header line."
    )


def _load_tags(path: str | Path) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(k): np.asarray(v, dtype=np.int64) for k, v in payload.items()}
    if isinstance(payload, list):
        return {
            str(item["tag"]): np.asarray(item["nodes"], dtype=np.int64)
            for item in payload
        }
    raise MeshInputError(f"Unsupported tag file format: {path}")


def _load_json(path: Path, elem_type, index_base) -> MeshData:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MeshInputError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise MeshInputError("JSON mesh must be an object with 'nodes'/'elements'.")
    for key in ("nodes", "elements"):
        if key not in payload:
            raise MeshInputError(f"JSON mesh is missing the {key!r} key.")

    et = elem_type or payload.get("elem_type")
    if et is None:
        raise MeshInputError("JSON mesh has no 'elem_type'; pass --elem-type.")
    base = payload.get("index_base", 0) if index_base is None else index_base
    tags = payload.get("node_tags", {}) or {}

    data = MeshData(
        nodes=np.asarray(payload["nodes"], dtype=float),
        elements=np.asarray(payload["elements"]),
        elem_type=et,
        node_tags={str(k): np.asarray(v) for k, v in tags.items()},
        source=str(path),
    )
    _check_index_base(data, int(base))
    return data


def _load_npz(path: Path, elem_type, index_base) -> MeshData:
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - numpy raises several subclasses
        raise MeshInputError(f"Could not read NPZ archive {path}: {exc}") from exc

    with archive:
        files = set(archive.files)
        for key in ("nodes", "elements"):
            if key not in files:
                raise MeshInputError(f"NPZ mesh is missing the {key!r} array.")

        et = elem_type
        if et is None and "elem_type" in files:
            et = str(archive["elem_type"])
        if et is None:
            raise MeshInputError("NPZ mesh has no 'elem_type'; pass --elem-type.")

        base = index_base
        if base is None:
            base = int(archive["index_base"]) if "index_base" in files else 0

        tags = {
            name[4:]: np.asarray(archive[name], dtype=np.int64)
            for name in files
            if name.startswith("tag_")
        }

        data = MeshData(
            nodes=np.asarray(archive["nodes"], dtype=float),
            elements=np.asarray(archive["elements"]),
            elem_type=et,
            node_tags=tags,
            source=str(path),
        )

    _check_index_base(data, int(base))
    return data


def _check_index_base(data: MeshData, index_base: int) -> None:
    if index_base not in (0, 1):
        raise MeshInputError("index_base must be 0 or 1.")
    if index_base == 1:
        data.elements = np.asarray(data.elements) - 1
        data.node_tags = {
            tag: nodes - 1 for tag, nodes in data.node_tags.items()
        }


def load_mesh_data(
    source: str | Path | MeshData,
    *,
    elem_type=None,
    index_base: int | None = None,
) -> MeshData:
    """Read mesh data from a JSON/NPZ file or a directory.

    CSV needs two files, use :func:`load_nodes_elements` instead (or pass a
    directory containing ``nodes.csv`` and ``elements.csv``).
    """

    if isinstance(source, MeshData):
        return source

    path = Path(source)

    if path.is_dir():
        for name in ("mesh.json", "mesh.npz"):
            candidate = path / name
            if candidate.is_file():
                return load_mesh_data(
                    candidate, elem_type=elem_type, index_base=index_base
                )
        nodes_csv = path / "nodes.csv"
        elements_csv = path / "elements.csv"
        if nodes_csv.is_file() and elements_csv.is_file():
            tags_file = path / "node_tags.json"
            return load_nodes_elements(
                nodes_csv,
                elements_csv,
                elem_type=elem_type,
                index_base=index_base or 0,
                node_tags=tags_file if tags_file.is_file() else None,
            )
        raise MeshInputError(
            f"No mesh file found in {path} (expected mesh.json/mesh.npz or "
            "nodes.csv + elements.csv)."
        )

    if not path.is_file():
        raise MeshInputError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path, elem_type, index_base)
    if suffix == ".npz":
        return _load_npz(path, elem_type, index_base)
    if suffix in (".csv", ".txt", ".dat"):
        raise MeshInputError(
            f"{path} is a single table. Provide both nodal coordinates and "
            "connectivity via --nodes/--elements, or a directory with "
            "nodes.csv + elements.csv."
        )
    if suffix == ".msh":
        raise MeshInputError(
            "gmsh .msh files are loaded with load_mesh(), not load_mesh_data()."
        )

    raise MeshInputError(f"Unsupported mesh file extension {suffix!r}.")


def load_nodes_elements(
    nodes_path: str | Path,
    elements_path: str | Path,
    *,
    elem_type,
    index_base: int = 0,
    node_tags: str | Path | dict[str, np.ndarray] | None = None,
) -> MeshData:
    """Build ``MeshData`` from a nodal coordinate table and a connectivity table."""

    if elem_type is None:
        raise MeshInputError("elem_type is required for CSV input.")

    nodes = _read_table(nodes_path)
    elements = _read_table(elements_path)

    if node_tags is None:
        tags: dict[str, np.ndarray] = {}
    elif isinstance(node_tags, (str, Path)):
        tags = _load_tags(node_tags)
    elif isinstance(node_tags, dict):
        tags = {str(k): np.asarray(v, dtype=np.int64) for k, v in node_tags.items()}
    else:
        raise MeshInputError("node_tags must be a path or a dict.")

    data = MeshData(
        nodes=nodes,
        elements=elements,
        elem_type=elem_type,
        node_tags=tags,
        source=f"{nodes_path} + {elements_path}",
    )
    _check_index_base(data, int(index_base))
    return data


def load_mesh(
    source,
    *,
    elem_type=None,
    index_base: int | None = None,
    fix_orientation: bool = True,
    verbose: bool = False,
) -> Mesh:
    """Load a custom mesh (JSON/NPZ/CSV directory/MSH) and return an EasyFEA mesh."""

    if isinstance(source, Mesh):
        return source

    if isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".msh":
        return _load_msh(Path(source), verbose=verbose)

    data = load_mesh_data(source, elem_type=elem_type, index_base=index_base)
    return build_mesh(
        data, fix_orientation=fix_orientation, verbose=verbose
    )


def _load_msh(path: Path, *, verbose: bool = False) -> Mesh:
    from EasyFEA import Mesher

    if not path.is_file():
        raise MeshInputError(f"File not found: {path}")

    mesher = Mesher(gmshVerbosity=False, verbosity=False)
    mesh = mesher.Mesh_Import_mesh(str(path))

    if mesh.dim not in (2, 3):
        raise MeshInputError(
            f"{path} contains a {mesh.dim}D mesh; only 2D and 3D meshes are supported."
        )
    if verbose:
        info = mesh_info(mesh)
        measure_key = "volume" if mesh.dim == 3 else "area"
        print(
            f"[gmsh import] {info['elem_type']}: Nn={info['Nn']}, "
            f"Ne={info['Ne']}, {measure_key}={info[measure_key]:.6g}, "
            f"tags={info['tags']}"
        )
    return mesh


# ---------------------------------------------------------------------------
# writers / inspection
# ---------------------------------------------------------------------------


def save_mesh_data(data: MeshData, path: str | Path) -> Path:
    """Write ``MeshData`` to ``.json``, ``.npz`` or a directory (all formats)."""

    path = Path(path)

    if path.suffix == "" or path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        _write_json(data, path / "mesh.json")
        _write_npz(data, path / "mesh.npz")
        save_mesh_csv(data, path)
        return path

    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        _write_json(data, path)
    elif suffix == ".npz":
        _write_npz(data, path)
    elif suffix == ".csv":
        raise MeshInputError("Use save_mesh_csv() for CSV output.")
    else:
        raise MeshInputError(f"Unsupported output extension {suffix!r}.")
    return path


def _write_json(data: MeshData, path: Path) -> None:
    payload = {
        "elem_type": str(data.elem_type),
        "nodes": np.asarray(data.nodes).tolist(),
        "elements": np.asarray(data.elements, dtype=int).tolist(),
        "node_tags": {
            tag: np.asarray(nodes, dtype=int).tolist()
            for tag, nodes in data.node_tags.items()
        },
        "index_base": 0,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_npz(data: MeshData, path: Path) -> None:
    arrays = {
        "nodes": np.asarray(data.nodes, dtype=float),
        "elements": np.asarray(data.elements, dtype=np.int64),
        "elem_type": np.asarray(str(data.elem_type)),
        "index_base": np.asarray(0),
    }
    for tag, nodes in data.node_tags.items():
        arrays[f"tag_{tag}"] = np.asarray(nodes, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def save_mesh_csv(data: MeshData, folder: str | Path) -> None:
    """Write ``nodes.csv``, ``elements.csv`` and ``node_tags.json``."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    nodes = np.asarray(data.nodes, dtype=float)
    np.savetxt(
        folder / "nodes.csv",
        nodes,
        delimiter=",",
        header="x,y,z",
        comments="",
        fmt="%.17g",
    )
    elements = np.asarray(data.elements, dtype=np.int64)
    header = ",".join(f"n{i}" for i in range(elements.shape[1]))
    np.savetxt(folder / "elements.csv", elements, delimiter=",", header=header, comments="", fmt="%d")
    (folder / "node_tags.json").write_text(
        json.dumps(
            {
                tag: np.asarray(tag_nodes, dtype=int).tolist()
                for tag, tag_nodes in data.node_tags.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def available_tags(mesh: Mesh) -> list[str]:
    tags = set()
    for group_elem in mesh.dict_groupElem.values():
        tags.update(group_elem.nodeTags)
    return sorted(tags)


def mesh_info(mesh: Mesh) -> dict:
    if mesh.dim not in (2, 3):
        raise MeshInputError(
            f"Only 2D and 3D meshes are supported, got dim={mesh.dim}."
        )
    try:
        group_elem = mesh.groupElem
    except Exception as exc:  # noqa: BLE001 - AmbiguousGroupError
        raise MeshInputError(
            "The mesh contains several element groups of the main dimension; "
            "a custom mesh must use a single 2D/3D element type."
        ) from exc

    quality = mesh.Get_Quality("aspect")
    quality_min = None
    if quality is not None and np.size(quality) > 0:
        quality_min = float(np.nanmin(quality))

    info = {
        "elem_type": str(group_elem.elemType),
        "dim": int(mesh.dim),
        "Nn": int(mesh.Nn),
        "Ne": int(mesh.Ne),
        "quality_min": quality_min,
        "orphans": len(mesh.orphanNodes),
        "tags": available_tags(mesh),
    }
    if mesh.dim == 2:
        info["area"] = float(mesh.area)
    else:
        info["volume"] = float(mesh.volume)
    return info
