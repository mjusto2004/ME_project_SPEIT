"""Command-line interface: ``me-project-speit {generate,train,show,list}``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .datasets import available_kinds

DEFAULT_DATA_DIR = Path("data")


def build_parser() -> argparse.ArgumentParser:
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
        "--params",
        nargs="+",
        default=["T0", "P0", "h"],
        help="simulation parameters to vary (inputs of the dataset); others keep defaults",
    )

    gen.add_argument(
        "--kind",
        "--layout",
        dest="kind",
        choices=available_kinds(),
        default="field",
        help="dataset kind. field: one row per simulation (params -> whole field); "
        "pointwise: one row per node (params, x, y, z -> temperature)",
    )

    tr = sub.add_parser("train", help="train a model on a dataset")
    tr.add_argument("--dataset", required=True, help="CSV path or name of a CSV in --data-dir")
    tr.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    tr.add_argument("--model", default="mlp", help="registered model name (see 'list models')")
    tr.add_argument("--hidden", type=int, nargs="+", default=None, help="hidden layer sizes (mlp)")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--patience", type=int, default=30)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    tr.add_argument("--output-dir", type=Path, default=None)

    show = sub.add_parser("show", help="plot stored simulations in 3D")
    show.add_argument("--dataset", default="data_test", help="CSV path or name in --data-dir")
    show.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    show.add_argument("--ids", type=int, nargs="+", default=[0, 1])

    ls = sub.add_parser("list", help="list available models or datasets")
    ls.add_argument("what", choices=["models", "datasets"])
    ls.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    return parser


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
    parser = build_parser()
    args = parser.parse_args(argv)

    # Heavy imports are deferred so each command only needs its own dependencies.
    if args.command == "generate":
        from .datasets import save_dataset

        try:
            save_dataset(
                args.n_samples, folder=args.output_dir, seed=args.seed, input_names=args.params, kind=args.kind
            )
        except ValueError as exc:
            parser.error(str(exc))

    elif args.command == "train":
        from .models import IncompatibleError
        from .training import TrainConfig, train

        try:
            cfg = TrainConfig(
                dataset=args.dataset,
                model=args.model,
                model_kwargs={"hidden": args.hidden} if args.hidden else {},
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                seed=args.seed,
                device=args.device,
                output_dir=args.output_dir,
            )
            train(cfg)
        except (ValueError, FileNotFoundError, IncompatibleError) as exc:
            parser.exit(2, f"error: {exc}\n")

    elif args.command == "show":
        from .datasets import resolve_dataset
        from .visualization import plot_fields

        plot_fields(resolve_dataset(args.dataset, args.data_dir), ids=args.ids)

    elif args.command == "list":
        _list(args.what, args.data_dir)


if __name__ == "__main__":
    main()
