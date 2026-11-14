"""Graph datasets (variable meshes + node connections) and the GNN. No EasyFEA needed."""

import numpy as np
import pytest
import torch

from me_project_speit.datasets import (
    DatasetSpec,
    FieldDataset,
    GraphDataset,
    load_dataset,
)
from me_project_speit.datasets.graph import edges_from_connectivity
from me_project_speit.inference import Predictor
from me_project_speit.models import GNN, IncompatibleError, build_model, compatible_models
from me_project_speit.training import TrainConfig, train

SIZES = [30, 45, 38, 52, 41, 36, 60, 33]  # nodes of each synthetic simulation


def make_spec():
    return DatasetSpec(
        name="toy", input_names=["P0", "x", "y", "z", "heated"], n_outputs=1,
        mesh_id="variable", kind="graph", rows_per_sample=None,
    )


def make_graph_dataset(sizes=SIZES, seed=0):
    """Chains of nodes (i -- i+1), each simulation with its own length."""
    rng = np.random.default_rng(seed)
    nodes, edges = [], []
    for sim, n in enumerate(sizes):
        p = rng.uniform()
        coord = rng.uniform(size=(n, 3))
        heated = (np.arange(n) == 0).astype(float)
        temp = p * coord.sum(axis=1)
        nodes.append(np.column_stack([np.full(n, sim), np.full(n, p), coord, heated, temp]))
        a = np.arange(n - 1)
        edges.append(np.column_stack([np.full(2 * (n - 1), sim),
                                      np.concatenate([a, a + 1]), np.concatenate([a + 1, a])]))
    return GraphDataset._from_blocks(make_spec(), np.vstack(nodes), np.vstack(edges))


def test_edges_from_two_tetrahedra():
    # tetrahedra (0,1,2,3) and (1,2,3,4) share the triangle (1,2,3): 6 + 6 - 3 = 9 undirected edges
    src, dst = edges_from_connectivity(np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))
    directed = set(zip(src.tolist(), dst.tolist()))
    assert len(src) == 18 and len(directed) == 18  # both directions, no duplicates
    assert all((b, a) in directed for a, b in directed)  # symmetric
    assert all(a != b for a, b in directed)  # no self loops
    assert (0, 4) not in directed  # nodes 0 and 4 share no element


def test_save_load_roundtrip(tmp_path):
    ds = make_graph_dataset()
    path = ds.save(tmp_path / "toy.csv")
    assert (tmp_path / "toy_edges.npz").exists() and (tmp_path / "toy.json").exists()
    loaded = load_dataset(path)
    assert isinstance(loaded, GraphDataset) and len(loaded) == len(SIZES)
    np.testing.assert_allclose(loaded.X, ds.X, rtol=1e-5)
    np.testing.assert_array_equal(loaded.edge_src, ds.edge_src)
    assert loaded.spec.rows_per_sample is None


def test_missing_edges_file_is_reported(tmp_path):
    path = make_graph_dataset().save(tmp_path / "toy.csv")
    (tmp_path / "toy_edges.npz").unlink()
    with pytest.raises(ValueError, match="Missing edges file"):
        load_dataset(path)


def test_split_keeps_whole_simulations():
    ds = make_graph_dataset()
    s = ds.split(seed=0)
    sims = [set(ds.sim[idx]) for idx in (s.train, s.val, s.test)]
    assert not (sims[0] & sims[1] or sims[0] & sims[2] or sims[1] & sims[2])
    for idx, group in zip((s.train, s.val, s.test), sims):
        assert len(idx) == sum(SIZES[i] for i in group)  # every node of each simulation
        assert ds.count_simulations(idx) == len(group)


def test_batch_is_one_big_graph_with_shifted_edges():
    ds = make_graph_dataset()
    rows = np.arange(len(ds.X))
    loader = ds.make_loader(ds.X, ds.Y, rows, batch_size=3, shuffle=False)
    x, edge_index, y = next(iter(loader))
    n_batch = SIZES[0] + SIZES[1] + SIZES[2]
    assert x.shape == (n_batch, 5) and y.shape == (n_batch, 1)
    assert edge_index.shape == (2, 2 * sum(n - 1 for n in SIZES[:3]))
    assert int(edge_index.max()) == n_batch - 1  # shifted, and never outside the batch
    # the second simulation's first edge starts at its first node (offset = SIZES[0])
    first_of_second = 2 * (SIZES[0] - 1)
    assert int(edge_index[0, first_of_second]) == SIZES[0]
    # edges only connect nodes with the same parameter value, i.e. of the same simulation
    assert torch.equal(x[edge_index[0], 0], x[edge_index[1], 0])


def test_gnn_shapes_and_edge_dependence():
    model = build_model("gnn", make_spec(), hidden=16, layers=2)
    x = torch.randn(10, 5)
    chain = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    other = torch.tensor([[5, 6, 7, 8], [6, 7, 8, 9]])
    out = model(x, chain)
    assert out.shape == (10, 1)
    assert not torch.allclose(out, model(x, other))  # the connections matter


def test_kinds_and_models_are_matched():
    graph = make_spec()
    assert compatible_models(graph) == ["gnn"]
    assert "gnn" not in compatible_models(DatasetSpec("f", ["a"], 50))
    with pytest.raises(IncompatibleError, match="GraphDataset"):
        build_model("gnn", DatasetSpec("f", ["a"], 50))
    with pytest.raises(IncompatibleError, match=r"Compatible models: \['gnn'\]"):
        build_model("mlp", graph)
    with pytest.raises(IncompatibleError):
        build_model("coord_mlp", graph)


def test_field_kind_rejects_sponge_parameters():
    with pytest.raises(ValueError, match="Unknown input parameter"):
        FieldDataset.build_spec(("T0", "r"))
    assert "r" in GraphDataset.build_spec(("T0", "r")).input_names


def test_train_and_predict_on_graphs(tmp_path):
    path = make_graph_dataset().save(tmp_path / "toy.csv")
    result = train(TrainConfig(dataset=path, model="gnn", model_kwargs={"hidden": [16], "layers": 2},
                               epochs=3, batch_size=4, device="cpu", output_dir=tmp_path / "run"))
    assert np.isfinite(result["test_mae"])

    predictor = Predictor(result["checkpoint"])
    ds = load_dataset(path)
    a, b = ds.node_start[0], ds.node_stop[0]
    c, d = ds.edge_start[0], ds.edge_stop[0]
    out = predictor.predict(ds.X[a:b], np.stack([ds.edge_src[c:d], ds.edge_dst[c:d]]))
    assert out.shape == (b - a, 1)
    with pytest.raises(ValueError, match="edge_index"):
        predictor.predict(ds.X[a:b])


def test_plot_run_writes_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from me_project_speit.plotting import latest_run, plot_run

    # coordinates of the synthetic graphs must be named x, y, z for the mesh plot
    ds = make_graph_dataset()
    ds.spec.input_names = ["P0", "x", "y", "z", "heated"]
    path = ds.save(tmp_path / "toy.csv")
    train(TrainConfig(dataset=path, model="gnn", model_kwargs={"hidden": [8], "layers": 1},
                      epochs=2, batch_size=4, device="cpu", output_dir=tmp_path / "runs" / "r1"))
    assert latest_run(tmp_path / "runs") == tmp_path / "runs" / "r1"

    saved = plot_run(tmp_path / "runs" / "r1", sample=0, half=True, save=tmp_path / "fig")
    assert [p.name for p in saved] == ["fig_curves.png", "fig_mesh.png"]
    assert all(p.stat().st_size > 1000 for p in saved)
    with pytest.raises(ValueError, match="--sample"):
        plot_run(tmp_path / "runs" / "r1", sample=99, save=tmp_path / "fig2")
