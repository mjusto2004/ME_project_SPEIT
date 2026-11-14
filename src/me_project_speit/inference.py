"""Load a trained checkpoint and predict fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .datasets import DatasetSpec
from .models import build_model


class Predictor:
    """A trained model together with its dataset spec and normalisation.

    Parameters
    ----------
    checkpoint : str or Path
        ``best_model.pt`` written by :func:`~me_project_speit.training.train`.
    """

    def __init__(self, checkpoint: str | Path):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.spec = DatasetSpec.from_dict(ckpt["spec"])
        self.model = build_model(ckpt["model_name"], self.spec, **ckpt["model_kwargs"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._x_mean, self._x_std = ckpt["x_mean"], ckpt["x_std"]
        self._y_mean, self._y_std = ckpt["y_mean"], ckpt["y_std"]

    def predict(self, x, edge_index=None) -> np.ndarray:
        """Predict outputs for inputs ``x`` of shape ``(n_inputs,)`` or ``(n, n_inputs)``.

        Inputs must be in the order ``self.spec.input_names``. Returns an array
        of shape ``(n, n_outputs)`` in physical units.

        For a model of a ``graph`` dataset, ``x`` has one row per node of the
        mesh and ``edge_index`` (shape ``(2, n_edges)``, rows ``src`` and
        ``dst``, both directions) gives the connections between these nodes.
        """
        x = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
        if x.shape[1] != self.spec.n_inputs:
            raise ValueError(
                f"This model expects {self.spec.n_inputs} inputs "
                f"({', '.join(self.spec.input_names)}), got {x.shape[1]}."
            )
        inputs = [(x - self._x_mean) / self._x_std]
        if self.spec.kind == "graph":
            if edge_index is None:
                raise ValueError("This model needs 'edge_index': the connections between nodes.")
            inputs.append(torch.as_tensor(edge_index, dtype=torch.long))
        elif edge_index is not None:
            raise ValueError("'edge_index' is only used by models of graph datasets.")
        with torch.no_grad():
            y = self.model(*inputs)
        return (y * self._y_std + self._y_mean).numpy()
