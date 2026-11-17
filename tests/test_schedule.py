"""Learning-rate schedules of the training loop (--scheduler none / cosine / plateau)."""

import math

import numpy as np
import pytest

from me_project_speit.cli import main
from me_project_speit.models.base import LRSchedule
from me_project_speit.training import TrainConfig, train

from test_pipeline import make_csv


class FakeOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": None}]


def rates(schedule, steps):
    opt, out = FakeOptimizer(), []
    for _ in range(steps):
        schedule.apply(opt)
        out.append(opt.param_groups[0]["lr"])
    return out


def test_constant_by_default():
    assert rates(LRSchedule("none", 1e-3, epochs=5, steps_per_epoch=4), 20) == [1e-3] * 20


def test_cosine_decays_from_lr_to_min_lr():
    lr = rates(LRSchedule("cosine", 1e-3, epochs=10, steps_per_epoch=5, min_lr=1e-5), 50)
    assert lr[0] == pytest.approx(1e-3)
    assert lr[-1] == pytest.approx(1e-5)  # the last batch uses min_lr
    assert all(a >= b for a, b in zip(lr, lr[1:]))  # never increases
    assert lr[25] == pytest.approx(1e-5 + (1e-3 - 1e-5) * 0.5 * (1 + math.cos(math.pi * 25 / 49)))


def test_cosine_warmup_ramps_up_first():
    lr = rates(LRSchedule("cosine", 1e-3, epochs=10, steps_per_epoch=5, warmup=2), 50)
    assert lr[:10] == pytest.approx([1e-3 * (i + 1) / 10 for i in range(10)])
    assert max(lr) == pytest.approx(1e-3) and lr[-1] < 1e-4


def test_plateau_reduces_then_stops_at_min_lr():
    s = LRSchedule("plateau", 1e-3, epochs=100, steps_per_epoch=1, min_lr=2e-4,
                   plateau_patience=2, plateau_factor=0.5)
    assert not s.end_of_epoch(improved=False) and s.lr == 1e-3
    assert not s.end_of_epoch(improved=False) and s.lr == 5e-4      # 2 epochs without progress
    assert not s.end_of_epoch(improved=True)                          # progress resets the count
    assert not s.end_of_epoch(improved=False) and s.lr == 5e-4
    assert not s.end_of_epoch(improved=False) and s.lr == 2.5e-4
    s.end_of_epoch(improved=False)
    assert not s.end_of_epoch(improved=False) and s.lr == 2e-4       # clipped at min_lr
    s.end_of_epoch(improved=False)
    assert s.end_of_epoch(improved=False)                             # at min_lr, still stuck: stop


@pytest.mark.parametrize("bad", [
    {"scheduler": "step"}, {"scheduler": "cosine", "min_lr": 1.0}, {"scheduler": "cosine", "warmup": 50},
    {"scheduler": "plateau", "plateau_factor": 1.5}, {"scheduler": "plateau", "plateau_patience": 0},
])
def test_invalid_schedules_are_rejected_before_training(tmp_path, bad):
    with pytest.raises(ValueError):
        TrainConfig(dataset=tmp_path / "x.csv", epochs=10, **bad)


def test_cosine_run_logs_the_rate(tmp_path):
    csv = make_csv(tmp_path / "toy.csv")
    result = train(TrainConfig(dataset=csv, epochs=6, patience=100, device="cpu", scheduler="cosine",
                               min_lr=1e-5, output_dir=tmp_path / "run"))
    h = np.genfromtxt(result["history"], delimiter=",", names=True)
    assert h["lr"][0] == pytest.approx(1e-3) and all(np.diff(h["lr"]) < 0)


def test_cli_options_and_plot(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from me_project_speit.plotting import plot_run

    csv = make_csv(tmp_path / "toy.csv")
    main(["train", "--dataset", str(csv), "--epochs", "4", "--device", "cpu", "--scheduler", "plateau",
          "--plateau-patience", "1", "--plateau-factor", "0.1", "--min-lr", "1e-6",
          "--output-dir", str(tmp_path / "run")])
    assert "plateau: x0.1 after 1 epochs" in capsys.readouterr().out
    saved = plot_run(tmp_path / "run", save=tmp_path / "fig")
    assert saved[0].name == "fig_curves.png" and saved[0].stat().st_size > 1000


def test_pod_history_without_rate_still_plots(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    from me_project_speit.plotting import plot_run

    csv = make_csv(tmp_path / "toy.csv")
    train(TrainConfig(dataset=csv, model="pod", model_kwargs={"regressor": "ridge"}, device="cpu",
                      output_dir=tmp_path / "run"))
    assert plot_run(tmp_path / "run", save=tmp_path / "fig")
