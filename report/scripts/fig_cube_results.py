"""Result figures of the cube part: learning curve, extrapolation, error maps, training curves."""

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
from fig_datasets import CMAP, cube_grid, shot  # noqa: E402

from me_project_speit.evaluation import evaluate  # noqa: E402

FIG = Path("report/figures")
RES = Path("report/results")
RUNS = Path("runs/report")
TEST = Path("data/report/cube_independent_100.csv")
COLORS = {"mean": "#999999", "mlp": "#3b6ea5", "mlp_cos": "#173f6b", "pod_ridge": "#e4a02e",
          "pod_gp": "#c03a2b", "coord_mlp": "#5a9e4b"}
LABELS = {"mean": "mean field", "mlp": "MLP (constant lr)", "mlp_cos": "MLP (cosine lr)",
          "pod_ridge": "POD + ridge", "pod_gp": "POD + GP", "coord_mlp": "coordinate MLP"}


def learning_curve():
    rows = json.loads((RES / "cube_learning_curve.json").read_text())
    limits = json.loads((RES / "cube_limits.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for model in ("mean", "mlp", "mlp_cos", "pod_ridge", "pod_gp"):
        pts = sorted((r["n"], r["sim_MAE"], r.get("sim_MAE_lo"), r.get("sim_MAE_hi"))
                     for r in rows if r["model"] == model)
        if model in ("pod_gp", "pod_ridge"):
            name = model.split("_")[1]
            pts = sorted([(n, limits[f"pod_{name}_{n}"], None, None) for n in (10, 20, 45)] + pts)
        n, m = [p[0] for p in pts], [p[1] for p in pts]
        ax.loglog(n, m, "o-", color=COLORS[model], label=LABELS[model])
        lo = [p[2] for p in pts]
        if all(v is not None for v in lo):
            ax.fill_between(n, lo, [p[3] for p in pts], color=COLORS[model], alpha=0.15, lw=0)
    cm = RES / "cube_coord_mlp_360_independent.csv"
    if cm.exists():
        maes = [float(r["MAE"]) for r in csv.DictReader(cm.open())]
        ax.loglog([360], [np.mean(maes)], "s", color=COLORS["coord_mlp"], ms=8, label=LABELS["coord_mlp"])
    ax.set_xlabel("number of training simulations (70 % used for fitting)")
    ax.set_ylabel("MAE per simulation on 100 new simulations (°C)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "cube_learning_curve.png", dpi=200)
    plt.close(fig)


def extrapolation():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for name in ("mlp", "pod_ridge", "pod_gp"):
        d = np.loadtxt(RES / f"extrap_{name}.txt")
        d = d[np.argsort(d[:, 0])]
        ax.semilogy(d[:, 0], d[:, 1], "o", ms=4, color=COLORS[name if name != "mlp" else "mlp_cos"],
                    label=LABELS[name] if name != "mlp" else "MLP (cosine lr)")
    ax.axvspan(50, 100, color="#999999", alpha=0.15)
    ax.text(75, 2e-6, "h never seen\nin training", ha="center", va="bottom", fontsize=9, color="#444444")
    ax.set_xlabel("h of the test simulation (W·m⁻²·K⁻¹)")
    ax.set_ylabel("MAE of the simulation (°C)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cube_extrapolation.png", dpi=200)
    plt.close(fig)


def error_maps():
    evs = {name: evaluate(RUNS / f"cube_{name}_720", TEST) for name in ("mlp_cos", "pod_gp")}
    worst = evs["mlp_cos"].worst().sim
    r_mlp, r_gp = evs["mlp_cos"].get(worst), evs["pod_gp"].get(worst)
    lo, hi = r_mlp.true.min(), r_mlp.true.max()
    panels = [("true", r_mlp.true, CMAP, (lo, hi)), ("pred_mlp", r_mlp.pred, CMAP, (lo, hi)),
              ("err_mlp", r_mlp.error, "RdBu_r", None), ("err_gp", r_gp.error, "RdBu_r", None)]
    info = {"sim": worst, "params": r_mlp.params, "mlp_mae": r_mlp.mae, "mlp_max": r_mlp.max_abs,
            "gp_mae": r_gp.mae, "gp_max": r_gp.max_abs}
    for name, values, cmap, clim in panels:
        grid = cube_grid(values)
        clipped = grid.clip(normal="y", origin=(0.5, 0.5, 0.5), invert=True)
        scale = 1.0
        if clim is None:
            m = np.abs(values).max()
            info[f"{name}_clim"] = m
            if m < 0.01:  # show tiny errors in milli-degrees
                scale, m = 1000.0, m * 1000.0
                info[f"{name}_unit"] = "mK"
            clim = (-m, m)
        grid.point_data["T"] = values * scale
        clipped = grid.clip(normal="y", origin=(0.5, 0.5, 0.5), invert=True)
        p = pv.Plotter(window_size=(780, 620))
        p.set_background("white")
        p.add_mesh(clipped, scalars="T", cmap=cmap, clim=clim,
                   scalar_bar_args=dict(title="", vertical=True, position_x=0.86, position_y=0.15,
                                        height=0.7, width=0.05, color="black", fmt="%.2f",
                                        label_font_size=18))
        p.add_mesh(clipped.extract_feature_edges(feature_angle=30), color="#333333", line_width=1.5)
        shot(p, FIG / f"cube_errmap_{name}.png", 1.1)
    (RES / "cube_errmap.json").write_text(json.dumps(info, indent=2))


def training_curves():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, ls in (("mlp", "-"), ("mlp_cos", "-")):
        for n, c in ((90, "#9ecae1"), (720, COLORS[name])):
            rows = list(csv.DictReader((RUNS / f"cube_{name}_{n}" / "history.csv").open()))
            ep = [int(r["epoch"]) for r in rows]
            ax = axs[0 if name == "mlp" else 1]
            ax.semilogy(ep, [float(r["train_mse_scaled"]) for r in rows], ls, color=c, lw=1,
                        label=f"{n} sims, train")
            ax.semilogy(ep, [float(r["val_mse_scaled"]) for r in rows], "--", color=c, lw=1,
                        label=f"{n} sims, validation")
    axs[0].set_title("MLP, constant lr + early stopping", fontsize=10)
    axs[1].set_title("MLP, cosine lr, 1500 epochs", fontsize=10)
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (standardised)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "cube_training_curves.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    learning_curve()
    extrapolation()
    error_maps()
    training_curves()
