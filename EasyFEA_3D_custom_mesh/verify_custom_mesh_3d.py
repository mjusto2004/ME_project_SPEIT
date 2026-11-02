"""Verification of the 3D custom mesh input interface.

Run:  python verify_custom_mesh_3d.py
Exits with code 0 when every check passes.
"""

from __future__ import annotations

import env_compat  # noqa: F401  (must precede NumPy/EasyFEA)

import gmsh
import json
import shutil
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import custom_mesh as cm
from demo_meshes_3d import hollow_box_mesh, thermal2_mesh
from thermal_custom_mesh import resolve_bc_nodes, solve_thermal
from thermal2_custom_mesh import boundary_nodes, patch_test, solve_transient

PASSED: list[str] = []
FAILED: list[str] = []


@contextmanager
def _temp_dir():
    path = Path(__file__).resolve().parent / f".verify3d_{uuid.uuid4().hex[:8]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def check(condition: bool, message: str) -> None:
    if condition:
        PASSED.append(message)
        print(f"  [ok] {message}")
    else:
        FAILED.append(message)
        print(f"  [FAIL] {message}")


def _patch_all(mesh) -> float:
    nodes = np.arange(mesh.Nn, dtype=np.int64)
    simu = solve_thermal(
        mesh, [(nodes, lambda x, y, z: 2.0 * x + 3.0 * y + 4.0 * z)]
    )
    T = np.asarray(simu.thermal).ravel()
    coords = np.asarray(mesh.coord)
    exact = 2.0 * coords[:, 0] + 3.0 * coords[:, 1] + 4.0 * coords[:, 2]
    return float(np.max(np.abs(T - exact)))


def _bcs(mesh):
    return [
        (resolve_bc_nodes(mesh, "left"), 0.0),
        (resolve_bc_nodes(mesh, "right"), 40.0),
    ]


def _run_transient(mesh):
    simu, T = solve_transient(mesh, _bcs(mesh), tmax=0.5, steps=20, alpha=0.5)
    return simu, T


def _symmetry_error(mesh, T, scale=1.0) -> tuple[float, int]:
    coords = np.asarray(mesh.coord)
    width = 3.0 * scale
    lookup = {
        (round(float(x), 12), round(float(y), 12), round(float(z), 12)): i
        for i, (x, y, z) in enumerate(coords)
    }
    errors = []
    for i, (x, y, z) in enumerate(coords):
        j = lookup.get(
            (round(width - float(x), 12), round(float(y), 12), round(float(z), 12))
        )
        if j is not None:
            errors.append(abs(T[i] + T[j] - 40.0))
    return (float(np.max(errors)) if errors else float("nan"), len(errors))


# ---------------------------------------------------------------------------


def test_single_elements():
    print("\n[1] Single 3D elements (volume + linear patch test)")
    cases = {
        "TETRA4": (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [0, 1, 2, 3],
            1.0 / 6.0,
        ),
        "HEXA8": (
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            [0, 1, 2, 3, 4, 5, 6, 7],
            1.0,
        ),
        "PRISM6": (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [0, 1, 2, 3, 4, 5],
            0.5,
        ),
    }
    for et, (coords, conn, volume) in cases.items():
        mesh = cm.build_mesh(
            cm.MeshData(np.asarray(coords, float), np.asarray([conn]), et)
        )
        check(
            abs(float(mesh.volume) - volume) < 1e-12,
            f"{et}: volume == {volume:g} (got {mesh.volume:.15g})",
        )
        error = _patch_all(mesh)
        check(error < 1e-12, f"{et}: linear patch test (max error {error:.1e})")


def test_hollow_box():
    print("\n[2] Hand-built hollow box (HEXA8)")
    data = hollow_box_mesh(n=4, nz=2)
    mesh = cm.build_mesh(data)
    check(abs(float(mesh.volume) - 8.0) < 1e-12, "volume == 8")
    patch_error = patch_test(mesh)
    check(
        patch_error is not None and patch_error < 1e-9,
        f"3D patch test (max error {patch_error:.1e})",
    )

    simu, T = _run_transient(mesh)
    check(
        float(T.min()) >= -1e-10 and float(T.max()) <= 40.0 + 1e-10,
        f"transient temperature in [0,40] ({T.min():.6g}..{T.max():.6g})",
    )

    left = resolve_bc_nodes(mesh, "left")
    right = resolve_bc_nodes(mesh, "right")
    steady = solve_thermal(mesh, [(left, 0.0), (right, 40.0)])
    T_steady = np.asarray(steady.thermal).ravel()
    sym_error, pairs = _symmetry_error(mesh, T_steady, scale=1.0)
    check(
        sym_error < 1e-8,
        f"steady symmetry T(x)+T(3-x)=40 on {pairs} pairs "
        f"(max error {sym_error:.1e})",
    )

    flux_left = float(np.sum(steady.Calc_Reaction(left)))
    flux_right = float(np.sum(steady.Calc_Reaction(right)))
    balance = abs(flux_left + flux_right) / max(abs(flux_left), abs(flux_right), 1e-30)
    check(balance < 1e-8, f"steady flux balance (relative residual {balance:.1e})")


def test_thermal2_roundtrip():
    print("\n[3] Thermal2 geometry round-trip and transient comparison")
    for et in ("TETRA4", "TETRA10", "PRISM18"):
        data, builtin = thermal2_mesh(et)
        custom = cm.build_mesh(data)
        check(
            abs(float(custom.volume) - float(builtin.volume)) < 1e-12,
            f"{et}: custom volume == builtin ({custom.volume:.12g})",
        )
        patch_error = patch_test(custom)
        check(
            patch_error is not None and patch_error < 1e-9,
            f"{et}: 3D patch test (max error {patch_error:.1e})",
        )

        bcs = _bcs(custom)
        _, T_custom = solve_transient(custom, bcs, tmax=0.5, steps=20)
        _, T_builtin = solve_transient(builtin, bcs, tmax=0.5, steps=20)
        difference = float(np.max(np.abs(T_custom - T_builtin)))
        check(
            difference < 1e-10,
            f"{et}: transient T matches Thermal2 (max diff {difference:.1e})",
        )

    # JSON round-trip for the exact Thermal2 mesh
    data, builtin = thermal2_mesh("TETRA4")
    custom = cm.build_mesh(data)
    _, T_reference = solve_transient(custom, _bcs(custom), tmax=0.5, steps=20)
    with _temp_dir() as tmp:
        path = tmp / "thermal2_3d.json"
        cm.save_mesh_data(data, path)
        reloaded = cm.load_mesh(path)
        _, T_reload = solve_transient(reloaded, _bcs(reloaded), tmax=0.5, steps=20)
    difference = float(np.max(np.abs(T_reload - T_reference)))
    check(difference < 1e-12, f"Thermal2 JSON round-trip (max diff {difference:.1e})")


def test_formats():
    print("\n[4] JSON / NPZ / CSV round-trip (hollow box)")
    data = hollow_box_mesh(n=3, nz=2)
    mesh = cm.build_mesh(data)
    _, reference = _run_transient(mesh)

    with _temp_dir() as tmp:
        json_path = tmp / "box.json"
        npz_path = tmp / "box.npz"
        csv_dir = tmp / "csv_only"
        cm.save_mesh_data(data, json_path)
        cm.save_mesh_data(data, npz_path)
        cm.save_mesh_csv(data, csv_dir)

        loaded = {
            "json": cm.load_mesh(json_path),
            "npz": cm.load_mesh(npz_path),
            "csv": cm.load_mesh(csv_dir, elem_type="HEXA8"),
        }
        for name, m in loaded.items():
            check(
                m.Nn == mesh.Nn and m.Ne == mesh.Ne,
                f"{name}: Nn/Ne preserved ({m.Nn}/{m.Ne})",
            )
            _, T = _run_transient(m)
            difference = float(np.max(np.abs(T - reference)))
            check(
                difference < 1e-12,
                f"{name}: identical transient field (max diff {difference:.1e})",
            )


def test_orientation():
    print("\n[5] Negative-volume connectivity auto-fix")
    cases = {
        "TETRA4": (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [0, 1, 2, 3],
        ),
        "HEXA8": (
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ),
        "PRISM6": (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [0, 1, 2, 3, 4, 5],
        ),
    }
    for et, (coords, conn) in cases.items():
        elem_type = cm.as_elem_type(et)
        reversed_data = cm.MeshData(
            np.asarray(coords, float),
            np.asarray([conn], int)[:, cm._REVERSE[elem_type]],
            et,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mesh = cm.build_mesh(reversed_data)
        warned = any("negative volume" in str(item.message) for item in caught)
        check(warned, f"{et}: reversal warning emitted")
        check(
            _patch_all(mesh) < 1e-12,
            f"{et}: reversed mesh still passes the patch test",
        )


def test_invalid_inputs():
    print("\n[6] Invalid 3D input rejection")
    tet = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )

    cases = {
        "3D element with 2D nodes": cm.MeshData(tet[:, :2], [[0, 1, 2, 3]], "TETRA4"),
        "degenerate tetra": cm.MeshData(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float),
            [[0, 1, 2, 3]],
            "TETRA4",
        ),
        "wrong number of nodes per element": cm.MeshData(tet, [[0, 1, 2]], "TETRA4"),
    }

    # mixed positive/negative orientation: two disjoint tets
    mixed_nodes = np.vstack([tet, tet + np.array([2.0, 0.0, 0.0])])
    mixed_conn = np.array([[0, 1, 2, 3], [4, 6, 5, 7]])
    cases["mixed orientation"] = cm.MeshData(mixed_nodes, mixed_conn, "TETRA4")

    for name, data in cases.items():
        try:
            cm.build_mesh(data)
        except cm.MeshInputError:
            check(True, f"{name}: rejected")
        else:
            check(False, f"{name}: should have been rejected")


def test_msh_import():
    print("\n[7] gmsh 3D .msh import")
    with _temp_dir() as tmp:
        path = tmp / "box3d.msh"
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box3d")
        gmsh.model.occ.addBox(0, 0, 0, 2, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.35)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
        gmsh.finalize()

        mesh = cm.load_mesh(path)
        check(mesh.dim == 3, f"3D mesh imported (Nn={mesh.Nn}, Ne={mesh.Ne})")
        check(abs(float(mesh.volume) - 2.0) < 1e-10, "imported volume == 2")
        error = patch_test(mesh)
        check(
            error is not None and error < 1e-9,
            f"imported mesh patch test (max error {error:.1e})",
        )


def main() -> int:
    test_single_elements()
    test_hollow_box()
    test_thermal2_roundtrip()
    test_formats()
    test_orientation()
    test_invalid_inputs()
    test_msh_import()

    print(f"\n{'=' * 60}")
    print(f"PASSED {len(PASSED)} / {len(PASSED) + len(FAILED)} checks")
    if FAILED:
        print("FAILED checks:")
        for message in FAILED:
            print(f"  - {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
