"""Dataset kinds.

To add a kind, create a module here with a :class:`Dataset` subclass decorated
with ``@register_dataset("name")`` and import it below.
"""

from .base import (
    DEFAULT_DATA_DIR,
    Dataset,
    DatasetSpec,
    Split,
    available_kinds,
    get_dataset_class,
    list_datasets,
    load_dataset,
    load_spec,
    register_dataset,
    resolve_dataset,
    save_dataset,
    save_spec,
    split_indices,
)

# Importing the modules registers the kinds.
from .field import FieldDataset  # noqa: E402
from .pointwise import PointwiseDataset  # noqa: E402
from .graph import GraphDataset  # noqa: E402

__all__ = [
    "DEFAULT_DATA_DIR",
    "Dataset",
    "DatasetSpec",
    "FieldDataset",
    "GraphDataset",
    "PointwiseDataset",
    "Split",
    "available_kinds",
    "get_dataset_class",
    "list_datasets",
    "load_dataset",
    "load_spec",
    "register_dataset",
    "resolve_dataset",
    "save_dataset",
    "save_spec",
    "split_indices",
]
