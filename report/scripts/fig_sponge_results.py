"""Result figures of the sponge part: error vs mesh resolution, training curves, error maps."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from sponge_tools import grid, rebuild  # noqa: E402

from me_project_speit.evaluation import evaluate  # noqa: E402

FIG = Path("report/figures")
RES = Path("report/results")
RUNS = Path("runs/report")
D = Path("data/report")
pv.OFF_SCREEN = True
MODELS = {"gnn": ("GNN, 4 rounds", "#3b6ea5"), "gnn8": ("GNN, 8 rounds", "#173f6b"), "gno": ("GNO", "#c03a2b")}
SEEDS = {7: (11, 50), 12: (12, 20), 18: (13, 20), 25: (14, 12)}
RANGES = {"T0": (273.0, 303.0), "Phi0": (15000.0, 30000.0), "h": (50.0, 200.0), "r": (0.05, 0.3)}


def baseline():
    """Mean-temperature baseline on every test set (same constant as exp_sponge.py)."""
    results = json.loads((RES / "sponge_results.json").read_text())
    mean_T = float(np.loadtxt(D / "sponge_train_1000.csv", delimiter=",", usecols=10).mean())
    out = {}
    ev = evaluate(RUNS / "sponge_gnn")  # only to get the true fields of the test split
    out["split"] = float(np.mean([np.abs(r.true - mean_T).mean() for r in ev.results]))
    for res in SEEDS:
        d = np.loadtxt(D / f"sponge_test_res{res}.csv", delimiter=",", usecols=(0, 10))
        out[str(res)] = float(np.mean([np.abs(d[d[:, 0] == s, 1] - mean_T).mean() for s in np.unique(d[:, 0])]))
    results["mean"] = {k: {"sim_MAE": v} for k, v in out.items()}
    (RES / "sponge_results.json").write_text(json.dumps(results, indent=2))
    return results


def resolution_plot(results):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = [7, 12, 18, 25]
    ax.axvspan(4.5, 10.5, color="#999999", alpha=0.15)
    ax.text(7.5, 40, "training\nresolutions", ha="center", fontsize=8, color="#444444")
    ax.plot(xs, [results["mean"][str(x)]["sim_MAE"] for x in xs], "o--", color="#999999", label="mean temperature")
    for name, (label, c) in MODELS.items():
        if name not in results:
            continue
        r = results[name]
        y = [r[str(x)]["sim_MAE"] for x in xs]
        lo = [r[str(x)]["sim_MAE_lo"] for x in xs]
        hi = [r[str(x)]["sim_MAE_hi"] for x in xs]
        ax.errorbar(xs, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o-", color=c, label=label, capsize=3)
    ax.set_xlabel("mesh resolution res of the test sponges (element size 1/res)")
    ax.set_ylabel("MAE per simulation (°C)")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "sponge_resolution.png", dpi=200)
    plt.close(fig)


def training_curves():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for name, (label, c) in MODELS.items():
        path = RUNS / f"sponge_{name}" / "history.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        ep = [int(r["epoch"]) for r in rows]
        ax.plot(ep, [float(r["train_mse_scaled"]) for r in rows], "-", color=c, lw=1, label=f"{label}, train")
        ax.plot(ep, [float(r["val_mse_scaled"]) for r in rows], "--", color=c, lw=1, label=f"{label}, validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardised)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "sponge_training_curves.png", dpi=200)
    plt.close(fig)


def where_errors(name="gno"):
    rows = list(csv.DictReader((RES / f"sponge_{name}_split.csv").open()))
    r = np.array([float(x["r"]) for x in rows])
    h = np.array([float(x["h"]) for x in rows])
    mae = np.array([float(x["MAE"]) for x in rows])
    nmae = np.array([float(x["NMAE"]) for x in rows])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    sc = axs[0].scatter(h, r, c=mae, s=14 + 6 * mae, cmap="viridis")
    fig.colorbar(sc, ax=axs[0], label="MAE of the sponge (°C)")
    axs[0].set_xlabel("h (W·m⁻²·K⁻¹)")
    axs[0].set_ylabel("hole radius r (m)")
    rng = np.array([float(x["MAE"]) / max(float(x["NMAE"]), 1e-9) for x in rows])
    axs[1].scatter(rng, mae, s=10, color=MODELS[name][1])
    axs[1].set_xlabel("temperature range of the sponge (°C)")
    axs[1].set_ylabel("MAE (°C)")
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"sponge_where_{name}.png", dpi=200)
    plt.close(fig)
    corr = {"r": float(np.corrcoef(r, mae)[0, 1]), "h": float(np.corrcoef(h, mae)[0, 1]),
            "range": float(np.corrcoef(rng, mae)[0, 1]), "nmae_mean": float(nmae.mean())}
    (RES / f"sponge_where_{name}.json").write_text(json.dumps(corr, indent=2))


def error_maps(res=25, i=None):
    """True field and the error of each model on one test sponge, rebuilt to get its tetrahedra."""
    csvp = D / f"sponge_test_res{res}.csv"
    evs = {n: evaluate(RUNS / f"sponge_{n}", csvp) for n in MODELS if (RUNS / f"sponge_{n}").exists()}
    if i is None:  # median sponge of the GNO
        ranked = sorted(evs["gno"].results, key=lambda r: r.mae)
        i = ranked[len(ranked) // 2].sim
    seed, n = SEEDS[res]
    s = rebuild(seed, n, i, {**RANGES, "res": (res, res)})
    true = evs["gno"].get(i).true
    assert len(s["T"]) == len(true) and np.allclose(s["T"], true, atol=1e-3), "rebuilt sponge differs"
    info = {"res": res, "sim": i, "nodes": len(true), "params": evs["gno"].get(i).params,
            "Tmin": float(true.min()), "Tmax": float(true.max())}
    panels = [("true", true, "inferno", (true.min(), true.max()), "T (°C)")]
    m = max(np.abs(evs[k].get(i).error).max() for k in evs)
    m = min(m, 30.0)
    for k in evs:
        panels.append((f"err_{k}", evs[k].get(i).error, "RdBu_r", (-m, m), "error (°C)"))
        info[k] = {"mae": evs[k].get(i).mae, "max": evs[k].get(i).max_abs}
    for name, values, cmap, clim, title in panels:
        g = grid(s["coord"], s["connect"], T=values)
        p = pv.Plotter(window_size=(700, 700))
        p.set_background("white")
        p.add_mesh(g.clip(normal="y", origin=(0.5, 0.5, 0.5), invert=True), scalars="T", cmap=cmap, clim=clim,
                   show_scalar_bar=False, scalar_bar_args=dict(title="", vertical=True, position_x=0.86, position_y=0.15, height=0.7,
                                        width=0.05, color="black", fmt="%.1f", label_font_size=18))
        p.camera_position = "iso"
        p.camera.zoom(1.0)
        p.screenshot(str(FIG / f"sponge_map_res{res}_{name}.png"))
        p.close()
    (RES / f"sponge_map_res{res}.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    results = baseline()
    resolution_plot(results)
    training_curves()
    where_errors("gno")
    where_errors("gnn")
    error_maps(25)
    error_maps(7)
