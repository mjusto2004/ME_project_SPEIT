"""Load a trained checkpoint and predict fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .dataset import DatasetSpec
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

    def predict(self, x) -> np.ndarray:
        """Predict outputs for inputs ``x`` of shape ``(n_inputs,)`` or ``(n, n_inputs)``.

        Inputs must be in the order ``self.spec.input_names``. Returns an array
        of shape ``(n, n_outputs)`` in physical units.
        """
        x = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)
        if x.shape[1] != self.spec.n_inputs:
            raise ValueError(
                f"This model expects {self.spec.n_inputs} inputs "
                f"({', '.join(self.spec.input_names)}), got {x.shape[1]}."
            )
        with torch.no_grad():
            y = self.model((x - self._x_mean) / self._x_std)
        return (y * self._y_std + self._y_mean).numpy()
