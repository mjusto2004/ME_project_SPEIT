"""Abstract :class:`Dataset`, its description (:class:`DatasetSpec`) and registry.

A dataset is a CSV file (one block of rows per simulation) accompanied by a JSON
*sidecar* (same name, ``.json`` extension) holding a :class:`DatasetSpec`.
The spec names the dataset **kind**; each kind is a subclass of
:class:`Dataset` registered with :func:`register_dataset`.

To create a new kind of dataset, subclass :class:`Dataset` and implement the
three hooks :meth:`~Dataset.build_spec`, :meth:`~Dataset.simulation_rows` and
:meth:`~Dataset.fields`. Loading, saving, generating, splitting and the
compatibility checks with models are inherited.
"""

from __future__ import annotations

import datetime
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..simulations import config
from ..parallel import parallel_map

DEFAULT_DATA_DIR = Path("data")

_KINDS: dict[str, type["Dataset"]] = {}


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def register_dataset(kind: str):
    """Class decorator registering a :class:`Dataset` subclass under ``kind``."""

    def decorator(cls: type[Dataset]) -> type[Dataset]:
        if kind in _KINDS:
            raise ValueError(f"A dataset kind named '{kind}' is already registered.")
        cls.kind = kind
        _KINDS[kind] = cls
        return cls

    return decorator


def available_kinds() -> list[str]:
    return list(_KINDS)


def get_dataset_class(kind: str) -> type[Dataset]:
    try:
        return _KINDS[kind]
    except KeyError:
        raise ValueError(f"Unknown dataset kind '{kind}'. Available: {available_kinds()}") from None


# --------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    """Shape and meaning of a dataset (stored as the JSON sidecar).

    Attributes
    ----------
    name : str
        Human-readable name.
    input_names : list[str]
        Names of the input columns, in order (e.g. ``["T0", "P0", "h"]``).
    n_outputs : int
        Number of output values per row.
    output_kind : str
        What the outputs are (e.g. ``"nodal_temperature"``).
    mesh_id : str or None
        Identifier of the mesh the outputs live on; two datasets only share
        output semantics if they share this id.
    input_ranges : dict
        Sampling range ``(min, max)`` per simulation parameter, used when
        generating data.
    kind : str
        Registered dataset kind (see :func:`available_kinds`); decides which
        :class:`Dataset` subclass reads the file.
    rows_per_sample : int or None
        Number of consecutive CSV rows belonging to one simulation (``None``
        when it varies from one simulation to the next). Splits are made per
        simulation so none is shared between train and test.
    """

    name: str
    input_names: list[str]
    n_outputs: int
    output_kind: str = "nodal_temperature"
    mesh_id: str | None = None
    input_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    kind: str = "field"
    rows_per_sample: int | None = 1

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"Unknown dataset kind '{self.kind}'. Available: {available_kinds()}")

    @property
    def n_inputs(self) -> int:
        return len(self.input_names)

    @property
    def n_columns(self) -> int:
        return self.n_inputs + self.n_outputs

    def describe(self) -> str:
        return (
            f"[{self.kind}] {self.n_inputs} inputs ({', '.join(self.input_names)}) -> "
            f"{self.n_outputs} outputs ({self.output_kind}, mesh={self.mesh_id})"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["input_ranges"] = {k: list(v) for k, v in self.input_ranges.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        d = dict(d)
        if "layout" in d:  # old name of "kind"
            d.setdefault("kind", d["layout"])
            del d["layout"]
        d["input_ranges"] = {k: tuple(v) for k, v in d.get("input_ranges", {}).items()}
        return cls(**d)


def spec_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".json")


def save_spec(spec: DatasetSpec, csv_path: str | Path) -> Path:
    path = spec_path(csv_path)
    path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    return path


def load_spec(csv_path: str | Path) -> DatasetSpec:
    """Read the sidecar of ``csv_path``.

    A sidecar-less CSV is only accepted if it has the legacy 563-column
    layout (``T0, P0, h`` + 560 nodal temperatures).
    """
    from .field import legacy_spec

    csv_path = Path(csv_path)
    sidecar = spec_path(csv_path)
    if sidecar.exists():
        return DatasetSpec.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))

    with csv_path.open() as f:
        n_cols = len(f.readline().strip().split(","))
    legacy = legacy_spec()
    if n_cols == legacy.n_columns:
        return legacy
    raise ValueError(
        f"{csv_path.name} has {n_cols} columns and no description file "
        f"'{sidecar.name}'. Create it (see DatasetSpec) so models can check "
        "compatibility."
    )


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
@dataclass
class Split:
    """Row indices of the train / validation / test subsets."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_indices(
    n: int,
    seed: int,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    rows_per_sample: int = 1,
) -> Split:
    """Randomly split ``n`` simulations (at least one in test and val).

    The split is made per simulation, then expanded to CSV row indices, so all
    the rows of a simulation land in the same subset (no leakage between
    train and test when a simulation spans several rows).
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))

    def rows(sims):
        if rows_per_sample == 1:
            return sims
        return (sims[:, None] * rows_per_sample + np.arange(rows_per_sample)).ravel()

    return Split(
        test=rows(perm[:n_test]),
        val=rows(perm[n_test : n_test + n_val]),
        train=rows(perm[n_test + n_val :]),
    )


# --------------------------------------------------------------------------
# Abstract dataset
# --------------------------------------------------------------------------
class Dataset(ABC):
    """Inputs ``X`` ``(rows, n_inputs)``, outputs ``Y`` ``(rows, n_outputs)`` and their spec.

    Subclasses define one kind of dataset by implementing three hooks:

    - :meth:`build_spec`: the shape of the data (inputs, outputs, rows per simulation);
    - :meth:`simulation_rows`: the CSV rows produced by ONE simulation;
    - :meth:`fields`: the outputs as ``(n_simulations, n_nodes)`` (for plotting).

    Everything else (reading and writing files, generation loop, splitting,
    validation) is provided here.
    """

    kind: ClassVar[str] = ""  # set by @register_dataset

    #: Simulation parameters this kind can vary: name -> (min, max).
    param_ranges: ClassVar[dict] = config.PARAM_RANGES
    #: Parameters varied by default (``generate`` without ``--params``).
    default_params: ClassVar[tuple] = ("T0", "P0", "h")

    def __init__(self, X: np.ndarray, Y: np.ndarray, spec: DatasetSpec):
        self.X, self.Y, self.spec = X, Y, spec

    def __len__(self) -> int:
        """Number of simulations (not rows)."""
        return len(self.X) // self.spec.rows_per_sample

    # ---- hooks to implement ------------------------------------------------
    @classmethod
    @abstractmethod
    def build_spec(cls, input_names, name: str = "cube") -> DatasetSpec:
        """Spec of a dataset of this kind varying the simulation parameters ``input_names``.

        Use :meth:`check_params` to validate ``input_names``.
        """

    @classmethod
    @abstractmethod
    def simulation_rows(cls, spec: DatasetSpec, params: dict[str, float]) -> np.ndarray:
        """CSV rows of one simulation run with ``params``.

        Returns an array of shape ``(spec.rows_per_sample, spec.n_columns)``:
        inputs first, then outputs. Call
        :meth:`~me_project_speit.simulations.CubeSimulation.run` here (the
        ``graph`` kind does not use this hook: it runs
        :meth:`~me_project_speit.simulations.SpongeSimulation.run` in its own
        ``generate``).
        """

    @abstractmethod
    def fields(self) -> np.ndarray:
        """Outputs reshaped as ``(n_simulations, n_nodes)``."""

    # ---- helpers for subclasses --------------------------------------------
    @classmethod
    def check_params(cls, input_names) -> list[str]:
        """Validate simulation parameter names against ``cls.param_ranges``."""
        names = list(input_names)
        unknown = [n for n in names if n not in cls.param_ranges]
        if unknown:
            raise ValueError(
                f"Unknown input parameter(s) {unknown} for kind '{cls.kind}'; "
                f"available: {sorted(cls.param_ranges)}"
            )
        if len(set(names)) != len(names):
            raise ValueError("Duplicate input parameters.")
        return names

    @classmethod
    def ranges(cls, names) -> dict[str, tuple[float, float]]:
        return {n: cls.param_ranges[n] for n in names}

    # ---- shared behaviour --------------------------------------------------
    @classmethod
    def generate(
        cls,
        n: int,
        input_names=("T0", "P0", "h"),
        name: str = "cube",
        seed: int | None = None,
        verbose: bool = True,
        workers: int = 1,
    ) -> "Dataset":
        """Run ``n`` simulations with parameters drawn uniformly in their ranges.

        The simulations run in ``workers`` processes. The parameters are all
        drawn first, in this process, so the dataset is the same whatever
        the number of workers.
        """
        spec = cls.build_spec(input_names, name=name)
        rng = np.random.default_rng(seed)
        all_params = [
            {k: rng.uniform(*spec.input_ranges[k]) for k in spec.input_ranges} for _ in range(n)
        ]
        blocks = []
        results = parallel_map(cls.simulation_rows, [spec] * n, all_params, workers=workers)
        for i, rows in enumerate(results):
            rows = np.asarray(rows, dtype=float)
            if rows.shape != (spec.rows_per_sample, spec.n_columns):
                raise ValueError(
                    f"{cls.__name__}.simulation_rows returned shape {rows.shape}, expected "
                    f"{(spec.rows_per_sample, spec.n_columns)} (rows_per_sample, n_columns)"
                )
            blocks.append(rows)
            if verbose:
                print(f"Simulation {i + 1}/{n} done")
        return cls._from_rows(np.vstack(blocks), spec)

    @classmethod
    def _from_rows(cls, data: np.ndarray, spec: DatasetSpec) -> "Dataset":
        return cls(
            X=data[:, : spec.n_inputs].astype(np.float32),
            Y=data[:, spec.n_inputs :].astype(np.float32),
            spec=spec,
        )

    @classmethod
    def from_csv(cls, path: str | Path, spec: DatasetSpec) -> "Dataset":
        """Read and validate a CSV against ``spec``."""
        path = Path(path)
        data = np.loadtxt(path, delimiter=",", ndmin=2)

        if data.shape[1] != spec.n_columns:
            raise ValueError(
                f"{path.name}: expected {spec.n_columns} columns "
                f"({spec.n_inputs} inputs + {spec.n_outputs} outputs), got {data.shape[1]}"
            )
        if len(data) % spec.rows_per_sample:
            raise ValueError(
                f"{path.name}: {len(data)} rows is not a multiple of rows_per_sample "
                f"({spec.rows_per_sample})."
            )
        if len(data) // spec.rows_per_sample < 3:
            raise ValueError("At least 3 simulations are required.")
        if not np.isfinite(data).all():
            raise ValueError("Dataset contains NaN or infinite values.")
        return cls._from_rows(data, spec)

    def count_simulations(self, row_indices: np.ndarray) -> int:
        """Number of simulations the given rows belong to."""
        return len(row_indices) // self.spec.rows_per_sample

    def make_loader(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        row_indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        pin_memory: bool = False,
    ):
        """Iterable of batches ``(inputs..., target)`` over the given rows.

        ``X`` and ``Y`` are the (normalised) inputs and outputs of the whole
        dataset. The last element of a batch is the target, the others are
        passed to ``model(*inputs)``. Here a batch is ``(x, y)``: independent rows.
        """
        tensors = TensorDataset(
            torch.from_numpy(X[row_indices]), torch.from_numpy(Y[row_indices])
        )
        return DataLoader(
            tensors, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
        )

    def save(self, path: str | Path) -> Path:
        """Write the CSV and its JSON sidecar; returns the CSV path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, np.hstack([self.X, self.Y]), delimiter=",")
        save_spec(self.spec, path)
        return path

    def split(
        self, seed: int, test_frac: float = 0.15, val_frac: float = 0.15
    ) -> Split:
        """Train / validation / test row indices, split per simulation."""
        return split_indices(
            len(self), seed, test_frac, val_frac, rows_per_sample=self.spec.rows_per_sample
        )


# --------------------------------------------------------------------------
# Loading, generating and discovering files
# --------------------------------------------------------------------------
def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset; the class is chosen from the ``kind`` in its spec."""
    spec = load_spec(path)
    return get_dataset_class(spec.kind).from_csv(path, spec)


def save_dataset(
    n: int,
    folder: str | Path = DEFAULT_DATA_DIR,
    seed: int | None = None,
    input_names=None,
    kind: str = "field",
    workers: int = 1,
) -> Path:
    """Generate ``n`` simulations of ``kind``; write a timestamped CSV and its JSON sidecar.

    ``input_names=None`` varies the default parameters of the kind.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"dataset_{stamp}_{n}"
    dataset_class = get_dataset_class(kind)
    input_names = input_names or dataset_class.default_params
    dataset = dataset_class.generate(n, input_names, name=stem, seed=seed, workers=workers)
    path = dataset.save(Path(folder) / f"{stem}.csv")
    print(f"Dataset saved in: {path} ({dataset.spec.describe()})")
    return path


def resolve_dataset(name_or_path: str | Path, data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Return the CSV path for a file path or a bare dataset name in ``data_dir``."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    data_dir = Path(data_dir)
    for option in (data_dir / candidate, data_dir / f"{candidate}.csv"):
        if option.is_file():
            return option
    available = [p.stem for p in sorted(data_dir.glob("*.csv"))] if data_dir.is_dir() else []
    raise FileNotFoundError(
        f"Dataset '{name_or_path}' not found (looked in '{data_dir}'). "
        f"Available: {available or 'none'}"
    )


def list_datasets(data_dir: str | Path = DEFAULT_DATA_DIR):
    """List ``(csv_path, spec_or_None, error_or_None)`` for every CSV in ``data_dir``."""
    out = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        try:
            out.append((path, load_spec(path), None))
        except ValueError as exc:
            out.append((path, None, str(exc)))
    return out
