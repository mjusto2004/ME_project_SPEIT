"""The `gno` model (radius neighbourhoods, mesh-independent) and `generate --res`. No EasyFEA needed."""

import numpy as np
import pytest
import torch

from me_project_speit.cli import main
from me_project_speit.datasets import DatasetSpec, GraphDataset, load_dataset
from me_project_speit.evaluation import evaluate
from me_project_speit.inference import Predictor
from me_project_speit.models import build_model, compatible_models
from me_project_speit.models.gno import radius_neighbours, simulation_ids
from me_project_speit.training import TrainConfig, train

NAMES = ["P0", "res", "x", "y", "z", "heated"]


def make_spec():
    return DatasetSpec(
        name="toy", input_names=NAMES, n_outputs=1, mesh_id="variable", kind="graph",
        rows_per_sample=None, input_ranges={"P0": (0.0, 1.0), "res": (3, 6)},
    )


def make_dataset(resolutions=(3, 4, 5, 3, 4, 5, 6, 4), seed=0):
    """Regular grids of the unit cube (res + 1 points per edge), linked to their 6 neighbours."""
    rng = np.random.default_rng(seed)
    nodes, edges = [], []
    for sim, res in enumerate(resolutions):
        g = np.linspace(0, 1, res + 1)
        coord = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        n = len(coord)
        ids = np.arange(n).reshape(res + 1, res + 1, res + 1)
        pairs = np.vstack([np.stack([a.ravel(), b.ravel()], 1) for a, b in (
            (ids[:-1], ids[1:]), (ids[:, :-1], ids[:, 1:]), (ids[:, :, :-1], ids[:, :, 1:]))])
        p = rng.uniform()
        temp = p * coord[:, 0] + coord.sum(1) ** 2  # a smooth field
        heated = (coord[:, 0] == 0).astype(float)
        nodes.append(np.column_stack([np.full(n, sim), np.full(n, p), np.full(n, res), coord, heated, temp]))
        e = np.vstack([pairs, pairs[:, ::-1]])
        edges.append(np.column_stack([np.full(len(e), sim), e]))
    return GraphDataset._from_blocks(make_spec(), np.vstack(nodes), np.vstack(edges))


def batch_of(ds, sims):
    """Concatenated inputs and shifted edges of some simulations, like the training loader."""
    xs, srcs, dsts, offset = [], [], [], 0
    for s in sims:
        a, b = ds.node_start[s], ds.node_stop[s]
        c, d = ds.edge_start[s], ds.edge_stop[s]
        xs.append(ds.X[a:b])
        srcs.append(ds.edge_src[c:d] + offset)
        dsts.append(ds.edge_dst[c:d] + offset)
        offset += b - a
    return np.concatenate(xs), np.stack([np.concatenate(srcs), np.concatenate(dsts)])


def test_simulation_ids_from_edges():
    ds = make_dataset()
    x, edges = batch_of(ds, [0, 3, 6])
    ids = simulation_ids(len(x), torch.as_tensor(edges))
    sizes = [ds.node_stop[s] - ds.node_start[s] for s in (0, 3, 6)]
    assert ids.tolist() == np.repeat([0, 1, 2], sizes).tolist()


def test_radius_neighbours_stay_in_ball_and_simulation():
    torch.manual_seed(0)
    pos = torch.rand(300, 3)
    groups = torch.repeat_interleave(torch.tensor([0, 1, 2]), torch.tensor([100, 120, 80]))
    src, dst = radius_neighbours(pos, groups, radius=0.3, k=7, generator=None)
    assert ((pos[src] - pos[dst]).norm(dim=1) <= 0.3 + 1e-6).all()
    assert (groups[src] == groups[dst]).all() and (src != dst).all()
    assert torch.bincount(dst, minlength=300).max() <= 7
    # with a large k every node of the ball is taken
    src, dst = radius_neighbours(pos, groups, radius=0.3, k=1000, generator=None)
    d = torch.cdist(pos, pos) <= 0.3
    d &= groups[:, None] == groups[None, :]
    d.fill_diagonal_(False)
    assert len(src) == int(d.sum())


def test_gno_is_compatible_with_graphs_only():
    assert "gno" in compatible_models(make_spec())
    field = DatasetSpec("f", ["T0", "Phi0", "h"], 560, mesh_id="cube_560")
    assert "gno" not in compatible_models(field)


def test_res_is_ignored_and_evaluation_is_reproducible():
    ds = make_dataset()
    model = build_model("gno", ds.spec, hidden=16, layers=2, radius=0.4, max_neighbours=8).eval()
    x, edges = batch_of(ds, [1])
    x, edges = torch.as_tensor(x), torch.as_tensor(edges)
    out = model(x, edges)
    assert out.shape == (len(x), 1)
    assert torch.equal(out, model(x, edges))  # fixed draw in evaluation
    changed = x.clone()
    changed[:, NAMES.index("res")] += 50.0
    assert torch.equal(out, model(changed, edges))
    assert model.ignored_inputs == ["res"]

    model = build_model("gno", ds.spec, hidden=16, layers=2, exclude_inputs=[]).eval()
    assert not torch.equal(model(x, edges), model(changed, edges))


def test_coordinates_cannot_be_excluded():
    with pytest.raises(ValueError, match="cannot be excluded"):
        build_model("gno", make_spec(), exclude_inputs=["x"])


def test_train_coarse_infer_fine(tmp_path):
    coarse = make_dataset((3, 4, 3, 4, 3, 4, 3, 4)).save(tmp_path / "coarse.csv")
    fine = make_dataset((7, 8, 7), seed=1)
    fine.spec.input_ranges["res"] = (7, 8)
    fine = fine.save(tmp_path / "fine.csv")

    result = train(TrainConfig(
        dataset=coarse, model="gno", model_kwargs={"hidden": 16, "layers": 2, "radius": 0.4},
        epochs=3, batch_size=4, device="cpu", output_dir=tmp_path / "run",
    ))
    assert np.isfinite(result["test_mae"])

    ds = load_dataset(fine)
    x, edges = batch_of(ds, [0])
    assert Predictor(result["checkpoint"]).predict(x, edges).shape == (len(x), 1)

    ev = evaluate(tmp_path / "run", fine)
    assert len(ev.results) == 3 and np.isfinite(ev.metrics()["sim_MAE"])
    assert "res" not in ev.train_ranges  # ignored by the model: no out-of-range warning
    assert "res outside" not in ev.summary()


def test_train_cli_options(tmp_path):
    path = make_dataset().save(tmp_path / "toy.csv")
    main(["train", "--dataset", str(path), "--model", "gno", "--hidden", "8", "--layers", "1",
          "--radius", "0.5", "--max-neighbours", "4", "--eval-neighbours", "16",
          "--exclude-inputs", "res", "heated", "--epochs", "1", "--device", "cpu",
          "--output-dir", str(tmp_path / "run")])
    ckpt = torch.load(tmp_path / "run" / "best_model.pt", weights_only=True)
    assert ckpt["model_kwargs"] == {"hidden": 8, "layers": 1, "radius": 0.5, "max_neighbours": 4,
                                    "eval_neighbours": 16, "exclude_inputs": ["res", "heated"]}
    assert Predictor(tmp_path / "run" / "best_model.pt").model.ignored_inputs == ["res", "heated"]


def test_generate_ranges():
    spec = GraphDataset.with_ranges(GraphDataset.build_spec(), {"res": (8, 12)})
    assert spec.input_ranges["res"] == (8, 12)
    with pytest.raises(ValueError, match="not a varied parameter"):
        GraphDataset.with_ranges(GraphDataset.build_spec(["T0", "Phi0"]), {"res": (8, 12)})
    with pytest.raises(SystemExit):
        main(["generate", "-n", "1", "--kind", "graph", "--res", "8", "12", "14"])
