"""Model-specific options of ``train`` are declared by the model classes."""

import pytest

from me_project_speit import cli
from me_project_speit.models import SurrogateModel, get_model_class, register_model


def parse_all(*argv):
    model = cli._requested_model(list(argv))
    args = cli.build_parser(model).parse_args(["train", "--dataset", "d", *argv])
    model_class = get_model_class(args.model)
    return model_class.kwargs_from_cli(args), model_class.fit_kwargs_from_cli(args)


def parse(*argv):
    return parse_all(*argv)[0]


def test_gnn_options():
    assert parse("--model", "gnn", "--hidden", "16", "--layers", "2") == {"hidden": 16, "layers": 2}


def test_mlp_hidden_is_a_list():
    assert parse("--model", "mlp", "--hidden", "16", "16") == {"hidden": [16, 16]}


def test_nothing_given_keeps_constructor_defaults():
    assert parse() == {}
    assert parse("--model", "gnn") == {}


def test_option_of_another_model_is_rejected(capsys):
    with pytest.raises(SystemExit):
        parse("--model", "mlp", "--layers", "2")
    assert "--layers" in capsys.readouterr().err


@register_model("cli_toy")
class CliToy(SurrogateModel):
    supported_datasets = ()  # compatible with no dataset: does not disturb the other tests

    def __init__(self, n_inputs, n_outputs, n_heads=2):
        super().__init__()

    @classmethod
    def add_cli_arguments(cls, parser):
        parser.add_argument("--n-heads", type=int, help="attention heads")

    @classmethod
    def add_fit_arguments(cls, parser):
        parser.add_argument("--clip", type=float, help="gradient clipping")

    def forward(self, x):
        return x

    def fit(self, train_loader, val_loader, **kwargs):
        return 0


def test_fit_options_are_separate_from_model_options():
    model_kwargs, fit_kwargs = parse_all("--model", "cli_toy", "--n-heads", "4", "--clip", "0.5")
    assert model_kwargs == {"n_heads": 4}
    assert fit_kwargs == {"clip": 0.5}
    assert parse_all("--model", "cli_toy") == ({}, {})


def test_builtin_weight_decay_goes_to_fit_only():
    assert parse_all("--model", "gnn", "--layers", "2", "--weight-decay", "0.01") == (
        {"layers": 2}, {"weight_decay": 0.01})


def test_new_model_exposes_its_own_option_without_touching_the_cli():
    assert parse("--model", "cli_toy", "--n-heads", "4") == {"n_heads": 4}


@register_model("cli_clash")
class CliClash(CliToy):
    @classmethod
    def add_cli_arguments(cls, parser):
        parser.add_argument("--lr", type=float)


def test_clashing_option_gives_a_clear_error():
    with pytest.raises(SystemExit, match="cli_clash"):
        cli.build_parser("cli_clash")


def test_help_shows_the_real_defaults(capsys):
    with pytest.raises(SystemExit):
        cli.main(["train", "--dataset", "d", "--model", "gnn", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "number of message-passing rounds (default: 4)" in out
    assert "(default: 0.0)" in out  # weight decay, from _fit_loop
    with pytest.raises(SystemExit):
        cli.main(["train", "--dataset", "d", "--model", "mlp", "--help"])
    assert "(default: 128 256 256)" in " ".join(capsys.readouterr().out.split())
