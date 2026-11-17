"""Command-line interface: ``me-project-speit {generate,train,infer,show,plot,list}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .datasets import available_kinds
from .utils import default_workers

DEFAULT_DATA_DIR = Path("data")


def build_parser(model: str | None = None) -> argparse.ArgumentParser:
    """The parser; with ``model``, the options of ``train`` include those the model declares."""
    parser = argparse.ArgumentParser(
        prog="me-project-speit",
        description="Learn temperature fields from finite-element simulations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="run simulations and save them to a CSV dataset")
    gen.add_argument("-n", "--n-samples", type=int, default=90)
    gen.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="number of processes running simulations at the same time (default: %(default)s, "
        "the number of cores up to 8). The dataset does not depend on it.",
    )
    gen.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="simulation parameters to vary (inputs of the dataset); others keep defaults. "
        "Default depends on --kind",
    )

    gen.add_argument(
        "--res",
        type=int,
        nargs="+",
        metavar=("MIN", "MAX"),
        default=None,
        help="mesh resolution (divisions of the cube edge) of a graph dataset: one value, or a "
        "range MIN MAX drawn per simulation. Default: the range in simulations/config.py",
    )
    gen.add_argument(
        "--kind",
        "--layout",
        dest="kind",
        choices=available_kinds(),
        default="field",
        help="dataset kind. field: one row per simulation (params -> whole field); "
        "pointwise: one row per node (params, x, y, z -> temperature); "
        "graph: random sponge meshes with their connections (variable node count)",
    )

    tr = sub.add_parser("train", help="train a model on a dataset")
    tr.add_argument("--dataset", required=True, help="CSV path or name of a CSV in --data-dir")
    tr.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    tr.add_argument("--model", default="mlp", help="registered model name (see 'list models')")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--batch-size", type=int, default=None,
                    help="rows per batch (simulations per batch for graph datasets); default depends on the dataset kind")
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--patience", type=int, default=30)
    tr.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none",
                    help="learning-rate schedule. cosine: decay from --lr to --min-lr over --epochs "
                    "(after --warmup epochs); plateau: multiply by --plateau-factor after "
                    "--plateau-patience epochs without progress (default: %(default)s)")
    tr.add_argument("--warmup", type=int, default=0, help="cosine: epochs of linear warmup (default: %(default)s)")
    tr.add_argument("--min-lr", type=float, default=0.0,
                    help="cosine: final rate; plateau: lowest rate (default: %(default)s)")
    tr.add_argument("--plateau-patience", type=int, default=10,
                    help="plateau: epochs without progress before reducing the rate (default: %(default)s)")
    tr.add_argument("--plateau-factor", type=float, default=0.5,
                    help="plateau: factor applied to the rate (default: %(default)s)")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    tr.add_argument("--output-dir", type=Path, default=None)
    tr.add_argument("--test-dataset", default=None,
                    help="extra test set (CSV path or name in --data-dir), e.g. independent runs; "
                    "must have the same inputs and outputs as --dataset")
    if model is not None:
        _add_model_options(tr, model)

    show = sub.add_parser("show", help="plot stored simulations in 3D")
    show.add_argument("--dataset", default="data_test", help="CSV path or name in --data-dir")
    show.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    show.add_argument("--ids", type=int, nargs="+", default=[0, 1])

    plot = sub.add_parser("plot", help="plot a finished training run (curves, predictions)")
    plot.add_argument("--run", type=Path, default=None, help="run folder (default: the latest in runs/)")
    plot.add_argument("--sample", type=int, default=0, help="which test simulation to draw")
    plot.add_argument("--half", action="store_true", help="draw only the nodes with y <= 0.5 (see inside)")
    plot.add_argument("--save", default=None, help="save PNG files <SAVE>_curves.png / _mesh.png instead of opening windows")

    inf = sub.add_parser("infer", help="run a trained model on a dataset: errors and 3D plots")
    inf.add_argument("--run", type=Path, default=None,
                     help="run folder or best_model.pt (default: the latest in runs/)")
    inf.add_argument("--dataset", default=None,
                     help="CSV path or name in --data-dir (default: the training dataset of the run)")
    inf.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    inf.add_argument("--subset", choices=["auto", "all", "test"], default="auto",
                     help="simulations to infer on. auto: the test split on the training dataset, "
                     "every simulation on another one (default: %(default)s)")
    inf.add_argument("--sim", type=int, default=None,
                     help="simulation (number in the dataset) drawn in 3D (default: the worst one)")
    inf.add_argument("--half", action="store_true", help="draw only the nodes with y <= 0.5 (see inside)")
    inf.add_argument("--mesh", action="store_true",
                     help="also open interactive PyVista views of --sim on the cube mesh (field, pointwise)")
    inf.add_argument("--top", type=int, default=5, help="number of worst / best simulations listed")
    inf.add_argument("--csv", type=Path, default=None, help="write the errors of every simulation to this CSV")
    inf.add_argument("--save", default=None, help="save the figure to <SAVE>_inference.png instead of opening it")
    inf.add_argument("--no-plot", action="store_true", help="only print the errors")

    ls = sub.add_parser("list", help="list available models or datasets")
    ls.add_argument("what", choices=["models", "datasets"])
    ls.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    return parser


def _add_model_options(train_parser: argparse.ArgumentParser, model: str) -> None:
    """Add the options a model declares in ``add_cli_arguments`` to ``train``."""
    from .models import get_model_class

    try:
        model_class = get_model_class(model)
    except ValueError:
        return  # unknown model: train() reports it, listing the available ones
    group = train_parser.add_argument_group(f"{model} options")
    try:
        model_class.add_cli_arguments(group)
        model_class.add_fit_arguments(group)
    except argparse.ArgumentError as exc:
        sys.exit(f"error: the options of model '{model}' clash with those of 'train': {exc}")
    _show_defaults(model_class, group)


def _show_defaults(model_class, group) -> None:
    """Append "(default: ...)" to the help of each model option, read from the signatures.

    The options themselves default to ``None`` ("not given"); the real default is
    the one of the constructor or of ``fit``/``_fit_loop``, found here.
    """
    import inspect

    sources = [model_class.__init__, model_class.fit, model_class._fit_loop]
    for action in group._group_actions:
        for source in sources:
            parameter = inspect.signature(source).parameters.get(action.dest)
            if parameter is not None and parameter.default is not inspect.Parameter.empty:
                value = parameter.default
                text = " ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
                action.help = f"{action.help or ''} (default: {text.replace('%', '%%')})".strip()
                break


def _requested_model(argv: list[str]) -> str:
    """Value of ``--model`` in ``argv`` (default ``mlp``), read before the full parse."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--model", default="mlp")
    return peek.parse_known_args(argv)[0].model


def _list(what: str, data_dir: Path) -> None:
    from .datasets import list_datasets
    from .models import available_models, compatible_models

    if what == "models":
        for name in available_models():
            print(name)
        return

    entries = list_datasets(data_dir)
    if not entries:
        print(f"No dataset found in '{data_dir}'.")
    for path, spec, error in entries:
        if spec is None:
            print(f"{path.stem}\n    INVALID: {error}")
        else:
            print(f"{path.stem}\n    {spec.describe()}\n    compatible models: {compatible_models(spec)}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    # The options of `train` depend on the model, which is itself an option.
    parser = build_parser(_requested_model(argv) if argv[:1] == ["train"] else None)
    args = parser.parse_args(argv)

    # Heavy imports are deferred so each command only needs its own dependencies.
    if args.command == "generate":
        from .datasets import save_dataset

        try:
            ranges = None
            if args.res is not None:
                if len(args.res) > 2:
                    raise ValueError("--res takes one value or two (MIN MAX).")
                ranges = {"res": (args.res[0], args.res[-1])}
            save_dataset(
                args.n_samples, folder=args.output_dir, seed=args.seed, input_names=args.params, kind=args.kind,
                workers=args.workers, ranges=ranges,
            )
        except ValueError as exc:
            parser.error(str(exc))

    elif args.command == "train":
        from .models import IncompatibleError, get_model_class
        from .training import TrainConfig, train

        try:
            model_class = get_model_class(args.model)  # unknown name: ValueError listing the models
            cfg = TrainConfig(
                dataset=args.dataset,
                model=args.model,
                model_kwargs=model_class.kwargs_from_cli(args),
                fit_kwargs=model_class.fit_kwargs_from_cli(args),
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                scheduler=args.scheduler,
                warmup=args.warmup,
                min_lr=args.min_lr,
                plateau_patience=args.plateau_patience,
                plateau_factor=args.plateau_factor,
                seed=args.seed,
                device=args.device,
                output_dir=args.output_dir,
                test_dataset=args.test_dataset,
            )
            train(cfg)
        except (ValueError, FileNotFoundError, IncompatibleError) as exc:
            parser.exit(2, f"error: {exc}\n")

    elif args.command == "infer":
        from .datasets import resolve_dataset
        from .evaluation import evaluate, plot_evaluation, show_on_mesh
        from .models import IncompatibleError
        from .utils import latest_run

        try:
            run = args.run or latest_run()
            dataset = None if args.dataset is None else resolve_dataset(args.dataset, args.data_dir)
            ev = evaluate(run, dataset, subset=args.subset)
            print(ev.summary(top=args.top))
            if args.csv is not None:
                print(f"Saved: {ev.to_csv(args.csv)}")
            chosen = ev.worst() if args.sim is None else ev.get(args.sim)
            if not args.no_plot:
                for path in plot_evaluation(ev, sim=chosen.sim, half=args.half, save=args.save):
                    print(f"Saved: {path}")
            if args.mesh:
                show_on_mesh(chosen)
        except (FileNotFoundError, ValueError, IncompatibleError) as exc:
            parser.exit(2, f"error: {exc}\n")

    elif args.command == "show":
        from .datasets import resolve_dataset
        from .visualization import plot_fields

        try:
            plot_fields(resolve_dataset(args.dataset, args.data_dir), ids=args.ids)
        except NotImplementedError as exc:
            parser.exit(2, f"error: {exc}\n")

    elif args.command == "plot":
        from .plotting import plot_run
        from .utils import latest_run

        try:
            run = args.run or latest_run()
            print(f"Run: {run}")
            for path in plot_run(run, sample=args.sample, half=args.half, save=args.save):
                print(f"Saved: {path}")
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(2, f"error: {exc}\n")

    elif args.command == "list":
        _list(args.what, args.data_dir)


if __name__ == "__main__":
    main()
