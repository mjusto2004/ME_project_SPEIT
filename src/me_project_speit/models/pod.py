"""POD (Proper Orthogonal Decomposition) + regression surrogate.

Idea: the 560-node temperature field lives in a very low-dimensional space.
1. POD: Y ≈ mean + C @ modes, with C the k POD coefficients per simulation.
2. Regression: learn (T0, P, h) -> C with a small model.
3. Prediction: decode the predicted coefficients back to the full field.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


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
