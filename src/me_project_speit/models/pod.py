"""POD (Proper Orthogonal Decomposition) + regression surrogate.

The temperature field lives in a very low-dimensional space:

1. POD: ``Y ≈ mean + C @ modes``, with ``C`` the k POD coefficients of each simulation.
2. Regression: learn ``parameters -> C`` with a small model (``gp`` or ``ridge``).
3. Prediction: decode the predicted coefficients back to the full field.

The fit is a one-shot computation (SVD + scikit-learn), not a gradient descent.
Its result is stored as plain tensors (buffers), so the model is saved,
loaded and used by :class:`~me_project_speit.inference.Predictor` like any other.
It works on the normalised inputs and outputs given by ``train``.
"""

from __future__ import annotations

import warnings
from itertools import combinations_with_replacement

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV

from ..datasets import FieldDataset
from .base import SurrogateModel
from .registry import register_model

REGRESSORS = ("gp", "ridge")


def poly_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    """All monomials of degree 1..``degree`` of the columns of ``x``."""
    columns = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(x.shape[1]), d):
            columns.append(x[:, list(combo)].prod(dim=1))
    return torch.stack(columns, dim=1)


@register_model("pod")
class POD(SurrogateModel):
    """POD of the fields + regression of the coefficients.

    Parameters
    ----------
    n_modes : int
        Number of POD modes kept.
    regressor : {"gp", "ridge"}
        ``gp``: one Gaussian process per mode (anisotropic RBF kernel); very
        accurate on smooth noise-free data. ``ridge``: ridge regression on
        polynomial features of the inputs.
    degree : int
        Degree of the polynomial features (``ridge`` only).
    """

    supported_datasets = (FieldDataset,)

    def __init__(self, n_inputs: int, n_outputs: int, n_modes: int = 5,
                 regressor: str = "gp", degree: int = 3):
        super().__init__()
        if regressor not in REGRESSORS:
            raise ValueError(f"Unknown regressor '{regressor}'. Available: {list(REGRESSORS)}")
        if n_modes < 1 or degree < 1:
            raise ValueError("n_modes and degree must be positive")
        self.n_inputs, self.n_outputs = n_inputs, n_outputs
        self.n_modes, self.regressor, self.degree = n_modes, regressor, degree

        def buffer(name, *shape):
            self.register_buffer(name, torch.zeros(*shape, dtype=torch.float64))

        buffer("mean", n_outputs)
        buffer("modes", n_modes, n_outputs)
        if regressor == "gp":
            buffer("gp_x", 0, n_inputs)  # training inputs; their number is known after fit
            buffer("gp_alpha", n_modes, 0)
            buffer("gp_length", n_modes, n_inputs)
            buffer("gp_amplitude", n_modes)
            buffer("gp_mean", n_modes)
            buffer("gp_std", n_modes)
        else:
            n_features = poly_features(torch.zeros(1, n_inputs), degree).shape[1]
            buffer("feat_mean", n_features)
            buffer("feat_std", n_features)
            buffer("coef", n_features, n_modes)
            buffer("intercept", n_modes)

    @classmethod
    def add_cli_arguments(cls, parser):
        parser.add_argument("--n-modes", type=int, help="number of POD modes")
        parser.add_argument("--regressor", choices=REGRESSORS, help="regression of the POD coefficients")
        parser.add_argument("--degree", type=int, help="polynomial degree (ridge only)")

    # ---- prediction ----------------------------------------------------------
    def _coefficients(self, x: torch.Tensor) -> torch.Tensor:
        if self.regressor == "ridge":
            f = (poly_features(x, self.degree) - self.feat_mean) / self.feat_std
            return f @ self.coef + self.intercept
        columns = []
        for k in range(self.n_modes):
            scale = self.gp_length[k]
            d2 = torch.cdist(x / scale, self.gp_x / scale).square()
            kernel = self.gp_amplitude[k] * torch.exp(-0.5 * d2)
            columns.append((kernel @ self.gp_alpha[k]) * self.gp_std[k] + self.gp_mean[k])
        return torch.stack(columns, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        field = self.mean + self._coefficients(x.to(self.mean.dtype)) @ self.modes
        return field.to(x.dtype)

    # ---- fitting -------------------------------------------------------------
    @staticmethod
    def _collect(loader):
        x, y = zip(*[(b[0], b[-1]) for b in loader])
        return torch.cat(x).double().numpy(), torch.cat(y).double().numpy()

    def _set(self, **values):
        device = self.mean.device
        for name, value in values.items():
            setattr(self, name, torch.as_tensor(np.asarray(value), dtype=torch.float64, device=device))

    def fit(self, train_loader, val_loader, *, device, on_epoch=None, on_best=None, y_std=None, **ignored):
        """Fit on the training set (one shot); the validation set is only reported.

        ``epochs``, ``lr`` and ``patience`` do not apply and are ignored.
        """
        X, Y = self._collect(train_loader)
        if len(X) <= self.n_modes:
            raise ValueError(f"Need more than n_modes={self.n_modes} training simulations, got {len(X)}")

        mean = Y.mean(axis=0)
        _, _, Vt = np.linalg.svd(Y - mean, full_matrices=False)
        modes = Vt[: self.n_modes]
        C = (Y - mean) @ modes.T
        self._set(mean=mean, modes=modes)

        with warnings.catch_warnings():
            # Hyper-parameters at their bounds are expected on noise-free FEM data.
            warnings.simplefilter("ignore", ConvergenceWarning)
            if self.regressor == "gp":
                self._fit_gp(X, C)
            else:
                self._fit_ridge(X, C)

        train_mse, _, _ = self.evaluate(train_loader, device, y_std)
        val_mse, val_mae, val_rmse = self.evaluate(val_loader, device, y_std)
        if on_epoch:
            on_epoch(1, train_mse, val_mse, val_mae, val_rmse)
        if on_best:
            on_best(1)
        print(f"POD ({self.n_modes} modes, {self.regressor}) | Train MSE: {train_mse:.6f} | "
              f"Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.4f}")
        return 1

    def _fit_gp(self, X, C):
        amplitude, length, alpha, c_mean, c_std = [], [], [], [], []
        for k in range(self.n_modes):
            mu, sigma = C[:, k].mean(), C[:, k].std() or 1.0
            kernel = (ConstantKernel(constant_value_bounds=(1e-3, 1e8))
                      * RBF(length_scale=[1.0] * self.n_inputs, length_scale_bounds=(1e-2, 1e3))
                      + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-12, 1e-1)))
            gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=5, random_state=0)
            gp.fit(X, (C[:, k] - mu) / sigma)
            product = gp.kernel_.k1  # Constant * RBF; the white noise only acts during the fit
            amplitude.append(product.k1.constant_value)
            length.append(np.broadcast_to(product.k2.length_scale, (self.n_inputs,)))
            alpha.append(gp.alpha_)
            c_mean.append(mu)
            c_std.append(sigma)
        self._set(gp_x=X, gp_alpha=np.array(alpha), gp_length=np.array(length),
                  gp_amplitude=amplitude, gp_mean=c_mean, gp_std=c_std)

    def _fit_ridge(self, X, C):
        F = poly_features(torch.from_numpy(X), self.degree).numpy()
        f_mean, f_std = F.mean(axis=0), F.std(axis=0)
        f_std = np.where(f_std < 1e-12, 1.0, f_std)  # constant features (a fixed parameter)
        ridge = RidgeCV(alphas=np.logspace(-8, 2, 21)).fit((F - f_mean) / f_std, C)
        self._set(feat_mean=f_mean, feat_std=f_std, coef=ridge.coef_.T, intercept=ridge.intercept_)

    # ---- loading -------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The number of GP training points is only known after fit: resize first.
        for name, buf in self._buffers.items():
            saved = state_dict.get(prefix + name)
            if saved is not None and saved.shape != buf.shape:
                self._buffers[name] = torch.zeros_like(saved, dtype=buf.dtype)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
