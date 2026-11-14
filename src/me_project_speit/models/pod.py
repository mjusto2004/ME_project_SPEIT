"""POD (Proper Orthogonal Decomposition) + regression surrogate.

Idea: the 560-node temperature field lives in a very low-dimensional space.
1. POD: Y ≈ mean + C @ modes, with C the k POD coefficients per simulation.
2. Regression: learn (T0, P, h) -> C with a small model.
3. Prediction: decode the predicted coefficients back to the full field.
"""

import pickle
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ..datasets import FieldDataset
from .base import SurrogateModel
from .registry import register_model


class POD:
    """Truncated POD computed with an SVD of the centered snapshots."""

    def __init__(self, n_modes=5):
        self.n_modes = n_modes

    def fit(self, Y):
        self.mean = Y.mean(axis=0)
        _, s, Vt = np.linalg.svd(Y - self.mean, full_matrices=False)
        self.modes = Vt[:self.n_modes]                      # (k, n_nodes)
        self.explained_variance = (s**2 / np.sum(s**2))[:self.n_modes]
        return self

    def encode(self, Y):
        return (Y - self.mean) @ self.modes.T               # (n, k)

    def decode(self, C):
        return self.mean + C @ self.modes                   # (n, n_nodes)


def physics_features(X):
    """Features suggested by the physics: T is linear in T0 and P for a fixed h."""
    T0, P, h = X.T
    return np.column_stack([T0, P, h, P / h, T0 / h, 1 / h, P * h, T0 * h, h**2])


def make_regressor(name):
    """Regressor mapping (T0, P, h) -> POD coefficients."""
    if name == "gp":
        # Anisotropic RBF (one length scale per input) + noise term, one GP per mode
        # (FEM data is noise-free and very smooth: hyperparameters may hit their bounds, that's fine)
        kernel = ConstantKernel(constant_value_bounds=(1e-3, 1e8)) \
            * RBF(length_scale=[1.0] * 3, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-12, 1e-1))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=5, random_state=0)
        return make_pipeline(StandardScaler(), MultiOutputRegressor(gp))
    if name == "ridge":
        # Linear model on physics-based features, regularization picked by cross-validation
        return make_pipeline(FunctionTransformer(physics_features), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-8, 2, 21)))
    raise ValueError(f"Unknown regressor: {name}")


class PODRegressor:
    """Full surrogate: (T0, P, h) -> temperature field (n_nodes,)."""

    def __init__(self, n_modes=5, regressor="gp"):
        self.pod = POD(n_modes)
        self.regressor_name = regressor
        self.regressor = make_regressor(regressor)

    def fit(self, X, Y):
        C = self.pod.fit(Y).encode(Y)
        self.regressor.fit(X, C)
        return self

    def predict(self, X):
        return self.pod.decode(self.regressor.predict(X))


# ---------------------------------------------------------------------------
# Integration in the project: a registered model built on the classes above
# (POD, physics_features, make_regressor, PODRegressor are used unchanged).
#
# `train` hands over the normalisation statistics (set_scaling), so PODRegressor
# gets the raw inputs and fields it expects. The fitted regressor is pickled into
# a byte buffer to travel in the checkpoint's state_dict. Limits: not
# differentiable, CPU only; load checkpoints only from a trusted source and with
# the same scikit-learn version.
# ---------------------------------------------------------------------------
@register_model("pod")
class PODSurrogate(SurrogateModel):
    """POD of the fields + regression of the coefficients (built on :class:`PODRegressor`).

    Parameters
    ----------
    n_modes : int
        Number of POD modes kept.
    regressor : {"gp", "ridge"}
        Regression of the coefficients: Gaussian processes or ridge on physics features.
    """

    supported_datasets = (FieldDataset,)

    def __init__(self, n_inputs: int, n_outputs: int, n_modes: int = 5, regressor: str = "gp"):
        super().__init__()
        if n_modes < 1:
            raise ValueError("n_modes must be positive")
        self.n_modes, self.regressor_name = n_modes, regressor
        PODRegressor(n_modes, regressor)  # fails now on an unknown regressor
        self.regressor: PODRegressor | None = None
        for name, size in (("x_mean", n_inputs), ("x_std", n_inputs),
                           ("y_mean", n_outputs), ("y_std", n_outputs)):
            self.register_buffer(name, torch.zeros(size) + (1.0 if name.endswith("std") else 0.0))
        self.register_buffer("blob", torch.zeros(0, dtype=torch.uint8))  # pickled regressor

    @classmethod
    def add_cli_arguments(cls, parser):
        parser.add_argument("--n-modes", type=int, help="number of POD modes")
        parser.add_argument("--regressor", choices=["gp", "ridge"], help="regression of the POD coefficients")

    def set_scaling(self, x_mean, x_std, y_mean, y_std) -> None:
        for name, value in (("x_mean", x_mean), ("x_std", x_std), ("y_mean", y_mean), ("y_std", y_std)):
            getattr(self, name).copy_(torch.as_tensor(value))

    def _raw(self, x, mean, std):
        return x.detach().cpu().double().numpy() * std.double().cpu().numpy() + mean.double().cpu().numpy()

    def fit(self, train_loader, val_loader, *, device, y_std, on_epoch=None, on_best=None, **ignored):
        """Fit on the training set (one shot); the validation set is only reported.

        ``epochs``, ``lr`` and ``patience`` do not apply and are ignored.
        """
        x, y = zip(*[(b[0], b[-1]) for b in train_loader])
        X = self._raw(torch.cat(x), self.x_mean, self.x_std)
        Y = self._raw(torch.cat(y), self.y_mean, self.y_std)
        with warnings.catch_warnings():
            # Hyper-parameters at their bounds are expected on noise-free FEM data.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.regressor = PODRegressor(self.n_modes, self.regressor_name).fit(X, Y)
        self._store()

        train_mse, _, _ = self.evaluate(train_loader, device, y_std)
        val_mse, val_mae, val_rmse = self.evaluate(val_loader, device, y_std)
        if on_epoch:
            on_epoch(1, train_mse, val_mse, val_mae, val_rmse)
        if on_best:
            on_best(1)
        print(f"POD ({self.n_modes} modes, {self.regressor_name}) | Train MSE: {train_mse:.6f} | "
              f"Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.4f}")
        return 1

    def _store(self) -> None:
        data = np.frombuffer(pickle.dumps(self.regressor), dtype=np.uint8).copy()
        self.blob = torch.from_numpy(data).to(self.blob.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.regressor is None:
            if self.blob.numel() == 0:
                raise RuntimeError("The POD model is not fitted yet.")
            self.regressor = pickle.loads(self.blob.cpu().numpy().tobytes())
        raw = self.regressor.predict(self._raw(x, self.x_mean, self.x_std))
        scaled = (raw - self.y_mean.double().cpu().numpy()) / self.y_std.double().cpu().numpy()
        return torch.as_tensor(scaled, dtype=x.dtype, device=x.device)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        saved = state_dict.get(prefix + "blob")  # its size is only known after fit
        if saved is not None:
            self.blob = torch.zeros_like(saved)
            self.regressor = None  # unpickled again on the next forward
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
