"""2D thermal simulation on a user supplied mesh.

This is ``examples/Thermal/Thermal1.py`` with the mesh block replaced by a
custom mesh input interface:

    Thermal1.py                              this script
    ---------------------------------------  ---------------------------------
    domain = Domain(...)                     mesh = load_mesh(...)   (JSON,
    circle = Circle(...)                     mesh = square_mesh(...)  NPZ, CSV,
    mesh = domain.Mesh_2D([circle], TRI6)    mesh = load_mesh(.msh)   MSH, ...)

Everything else (thermal model, Dirichlet BCs, solve) follows the example.

Examples
--------
Run the Thermal1 geometry through a JSON mesh (built with gmsh, then reloaded)::

    python thermal_custom_mesh.py --demo thermal1 --elem-type TRI6

Run a hand-built 16x16 TRI3 mesh of the unit square::

    python thermal_custom_mesh.py --demo square --n 16 --elem-type TRI3

Run your own mesh files::

    python thermal_custom_mesh.py --nodes nodes.csv --elements elements.csv \
        --elem-type TRI3 --bc left:0 --bc right:40

Boundary conditions are ``SPEC:VALUE`` where SPEC is a node tag (e.g. ``left``),
one of ``xmin/xmax/ymin/ymax``, or a coordinate cut such as ``x=0.5``.
The default is ``xmin:0`` and ``xmax:40``.
"""

from __future__ import annotations

import env_compat  # noqa: F401  (must precede NumPy/EasyFEA)

import argparse
import sys
from pathlib import Path

import numpy as np

from EasyFEA import Models, Simulations

import custom_mesh as cm
from demo_meshes import mesh_to_data, square_mesh, thermal1_mesh, u_shape_mesh

DEFAULT_BCS = [("xmin", 0.0), ("xmax", 40.0)]


# ---------------------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------------------


def _mesh_tolerance(mesh, tol: float | None) -> float:
    if tol is not None:
        if tol <= 0:
            raise ValueError("--tol must be positive.")
        return float(tol)
    extent = np.ptp(np.asarray(mesh.coord)[:, :3], axis=0)
    diagonal = float(np.linalg.norm(extent))
    return max(1e-12, 1e-9 * diagonal)


def resolve_bc_nodes(mesh, spec: str, tol: float | None = None) -> np.ndarray:
    """Return node indices for ``spec`` (tag, xmin/xmax/..., or x=.../z=...)."""

    spec = spec.strip()
    if not spec:
        raise ValueError("Empty boundary condition selector.")

    tags = cm.available_tags(mesh)
    if spec.startswith("tag:"):
        tag = spec[4:].strip()
        if tag not in tags:
            raise ValueError(
                f"Unknown tag {tag!r}. Available tags: {tags or 'none'}."
            )
        return np.asarray(mesh.Nodes_Tags(tag), dtype=np.int64)

    if spec in tags:
        return np.asarray(mesh.Nodes_Tags(spec), dtype=np.int64)

    coords = np.asarray(mesh.coord)[:, :3]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    atol = _mesh_tolerance(mesh, tol)

    if spec in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        values = {
            "xmin": x.min(),
            "xmax": x.max(),
            "ymin": y.min(),
            "ymax": y.max(),
            "zmin": z.min(),
            "zmax": z.max(),
        }
        axis = {"x": x, "y": y, "z": z}[spec[0]]
        return np.where(np.isclose(axis, values[spec], atol=atol))[0].astype(np.int64)

    for axis_name, axis in (("x", x), ("y", y), ("z", z)):
        prefix = f"{axis_name}="
        if spec.startswith(prefix):
            try:
                value = float(spec[len(prefix) :])
            except ValueError as exc:
                raise ValueError(f"Invalid coordinate in {spec!r}.") from exc
            return np.where(np.isclose(axis, value, atol=atol))[0].astype(np.int64)

    raise ValueError(
        f"Unknown boundary selector {spec!r}. Use a node tag, "
        "xmin/xmax/ymin/ymax/zmin/zmax or x=<value>/y=<value>/z=<value>. "
        f"Available tags: {tags or 'none'}."
    )


def parse_bcs(items: list[str] | None) -> list[tuple[str, float]]:
    """Parse ``--bc SPEC:VALUE`` options."""

    if not items:
        return list(DEFAULT_BCS)

    bcs = []
    for item in items:
        if ":" not in item:
            raise ValueError(
                f"Invalid --bc {item!r}: expected SPEC:VALUE, e.g. left:0."
            )
        spec, value = item.rsplit(":", 1)
        try:
            bcs.append((spec.strip(), float(value)))
        except ValueError as exc:
            raise ValueError(f"Invalid value in --bc {item!r}.") from exc
    return bcs


def solve_thermal(
    mesh,
    bcs_nodes: list[tuple[np.ndarray, float]],
    *,
    k: float = 1.0,
    c: float = 1.0,
    rho: float = 1.0,
    thickness: float = 1.0,
):
    """Assemble and solve the steady thermal problem (same as Thermal1)."""

    model = Models.Thermal(
        k=k, c=c, thickness=thickness if mesh.dim == 2 else 1.0
    )
    simu = Simulations.Thermal(mesh, model)
    simu.rho = rho

    for nodes, value in bcs_nodes:
        if nodes.size == 0:
            raise RuntimeError("A boundary condition selected zero nodes.")
        simu.add_dirichlet(nodes, [value], ["t"])

    simu.Solve()
    return simu


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def save_results(path: str | Path, mesh, temperature: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(mesh.coord)
    table = np.column_stack(
        [np.arange(mesh.Nn), coords[:, 0], coords[:, 1], temperature]
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="node,x,y,t",
        comments="",
        fmt=["%d", "%.17g", "%.17g", "%.17g"],
    )
    return path


def plot_temperature(path: str | Path, mesh, temperature: np.ndarray, simu, title: str):
#     """Save a Matplotlib contour plot of the nodal temperature."""

#     import matplotlib

#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     from matplotlib.tri import Triangulation

#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)

#     group_elem = mesh.Get_list_groupElem(2)[0]
#     connect = np.asarray(group_elem.connect)
#     topology = str(group_elem.elemType)

#     if topology.startswith("TRI"):
#         triangles = connect[:, :3]
#     elif topology.startswith("QUAD"):
#         triangles = np.vstack([connect[:, [0, 1, 2]], connect[:, [0, 2, 3]]])
#     else:
#         raise ValueError(f"Cannot plot element type {topology}.")

#     coords = np.asarray(mesh.coord)
#     triangulation = Triangulation(coords[:, 0], coords[:, 1], triangles)

#     fig, ax = plt.subplots(figsize=(7.2, 6.0))
#     contour = ax.tripcolor(
#         triangulation, np.asarray(temperature), shading="gouraud", cmap="inferno"
#     )
#     ax.set_aspect("equal")
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_title(title)
#     fig.colorbar(contour, ax=ax, label="Temperature")
#     fig.tight_layout()
#     fig.savefig(path, dpi=200)
#     plt.close(fig)
#     return path

        from EasyFEA import PyVista
        PyVista.Plot(simu, temperature, plotMesh=True, nodeValue=True).show()
        return "No Path"


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2D thermal simulation on a custom mesh (based on Thermal1)."
    )
    parser.add_argument(
        "--demo",
        choices=("thermal1", "square", "u"),
        default="thermal1",
        help="built-in demo geometry (default: thermal1)",
    )
    parser.add_argument(
        "--elem-type",
        default=None,
        help="element type: TRI3, TRI6, QUAD4, QUAD8 "
        "(default: TRI6 for thermal1, TRI3 for square, QUAD4 for u)",
    )
    parser.add_argument("--n", type=int, default=8,
                        help="square/U demo: cells per side (square) or per unit length (U)")
    parser.add_argument("--size", type=float, default=1.0,
                        help="thermal1/square demo: geometry size")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="U demo: geometry scale (base U is 3 x 2)")
    parser.add_argument("--mesh", default=None, help="mesh file: .json/.npz/.msh or CSV directory")
    parser.add_argument("--nodes", default=None, help="CSV/TXT nodal coordinates")
    parser.add_argument("--elements", default=None, help="CSV/TXT element connectivity")
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=None,
                        help="0-based (default) or 1-based connectivity")
    parser.add_argument("--tags", default=None, help="JSON file of node tags")
    parser.add_argument(
        "--bc",
        action="append",
        default=None,
        metavar="SPEC:VALUE",
        help="Dirichlet BC, e.g. left:0 or xmin:0 (repeatable)",
    )
    parser.add_argument("--k", type=float, default=1.0, help="thermal conductivity")
    parser.add_argument("--c", type=float, default=1.0, help="mass heat capacity")
    parser.add_argument("--rho", type=float, default=1.0, help="density")
    parser.add_argument("--thickness", type=float, default=1.0, help="2D thickness")
    parser.add_argument("--tol", type=float, default=None, help="coordinate tolerance")
    parser.add_argument("--export", default=None,
                        help="write the demo mesh to .json/.npz (or directory)")
    parser.add_argument("--save", nargs="?", const="auto", default=None,
                        help="save nodal results to CSV")
    parser.add_argument("--plot", nargs="?", const="auto", default=None,
                        help="save a temperature plot to PNG")
    return parser


def _run(args) -> int:
    if args.mesh and (args.nodes or args.elements):
        raise ValueError("Use either --mesh or --nodes/--elements, not both.")
    if bool(args.nodes) != bool(args.elements):
        raise ValueError("--nodes and --elements must be provided together.")

    builtin_mesh = None
    data = None
    name = "custom"

    if args.mesh:
        mesh = cm.load_mesh(
            args.mesh,
            elem_type=args.elem_type,
            index_base=args.index_base,
            verbose=True,
        )
        name = Path(args.mesh).stem
    elif args.nodes:
        elem_type = args.elem_type
        if elem_type is None:
            raise ValueError("--elem-type is required with --nodes/--elements.")
        data = cm.load_nodes_elements(
            args.nodes,
            args.elements,
            elem_type=elem_type,
            index_base=0 if args.index_base is None else args.index_base,
            node_tags=args.tags,
        )
        mesh = cm.build_mesh(data, verbose=True)
        name = Path(args.nodes).stem
    else:
        default_types = {"thermal1": "TRI6", "square": "TRI3", "u": "QUAD4"}
        elem_type = args.elem_type or default_types[args.demo]
        if args.demo == "thermal1":
            data, builtin_mesh = thermal1_mesh(elem_type, a=args.size)
        elif args.demo == "square":
            data = square_mesh(args.n, elem_type, size=args.size)
        else:
            data = u_shape_mesh(args.n, elem_type, scale=args.scale)
        mesh = cm.build_mesh(data, verbose=True)
        name = f"{args.demo}_{elem_type}"

    if mesh.dim != 2:
        raise ValueError(
            "thermal_custom_mesh.py is the 2D entry point; "
            "use thermal2_custom_mesh.py for 3D meshes."
        )

    if args.export:
        export_data = data if data is not None else mesh_to_data(mesh)
        cm.save_mesh_data(export_data, args.export)
        print(f"[export] {args.export}")

    info = cm.mesh_info(mesh)
    quality = info["quality_min"]
    quality_text = f"{quality:.4g}" if quality is not None else "n/a"
    print(
        f"\nMesh     : {info['elem_type']}, Nn={info['Nn']}, Ne={info['Ne']}, "
        f"area={info['area']:.8g}, min quality={quality_text}"
    )
    print(f"Tags     : {info['tags'] or 'none'}")

    bcs = parse_bcs(args.bc)
    bcs_nodes = []
    for spec, value in bcs:
        nodes = resolve_bc_nodes(mesh, spec, tol=args.tol)
        if nodes.size == 0:
            raise RuntimeError(f"Boundary condition {spec!r} selected no nodes.")
        bcs_nodes.append((nodes, value))
        print(f"BC       : {spec} -> {nodes.size} nodes, T={value:g}")

    simu = solve_thermal(
        mesh,
        bcs_nodes,
        k=args.k,
        c=args.c,
        rho=args.rho,
        thickness=args.thickness,
    )
    temperature = np.asarray(simu.thermal).ravel()

    print(
        f"Results  : Tmin={temperature.min():.8g}, "
        f"Tmax={temperature.max():.8g}, nodes={temperature.size}"
    )

    # --- verification / comparison -------------------------------------
    if builtin_mesh is not None:
        reference = solve_thermal(
            builtin_mesh,
            bcs_nodes,
            k=args.k,
            c=args.c,
            rho=args.rho,
            thickness=args.thickness,
        )
        reference_temperature = np.asarray(reference.thermal).ravel()
        if reference_temperature.size != temperature.size:
            raise RuntimeError("Built-in and custom meshes have different node counts.")
        difference = float(np.max(np.abs(temperature - reference_temperature)))
        print(f"Compare  : max |T_custom - T_EasyFEA| = {difference:.3e}")
        if difference > 1e-8:
            print("WARNING  : custom mesh and reference solution differ.", file=sys.stderr)

    if args.demo == "square" and not args.mesh and not args.nodes:
        if len(bcs) == 2 and {spec for spec, _ in bcs} == {"xmin", "xmax"}:
            values = dict(bcs)
            x = np.asarray(mesh.coord)[:, 0]
            xmin, xmax = float(x.min()), float(x.max())
            exact = values["xmin"] + (values["xmax"] - values["xmin"]) * (
                (x - xmin) / (xmax - xmin)
            )
            error = float(np.max(np.abs(temperature - exact)))
            print(f"Analytic : max |T - T_exact| = {error:.3e} (T is linear in x)")

    # --- outputs --------------------------------------------------------
    if args.save:
        path = (
            Path("results") / f"{name}_thermal.csv"
            if args.save == "auto"
            else Path(args.save)
        )
        saved = save_results(path, mesh, temperature)
        print(f"Saved    : {saved}")

    if args.plot:
        path = (
            Path("results") / f"{name}_thermal.png"
            if args.plot == "auto"
            else Path(args.plot)
        )
        saved = plot_temperature(path, mesh, temperature, simu, title=name)
        print(f"Plot     : {saved}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, cm.MeshInputError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
