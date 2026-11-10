"""The finite-element simulations that generate the datasets.

- :class:`CubeSimulation`: fixed cube with a cylindrical source (kinds ``field``
  and ``pointwise``);
- :class:`SpongeSimulation`: cube with random spherical holes (kind ``graph``).

Their constants and parameters are in :mod:`~me_project_speit.simulations.config`.
EasyFEA is only imported when a simulation runs.
"""

from . import config
from .base import Simulation
from .cube import CubeSimulation
from .sponge import SpongeSimulation

__all__ = ["CubeSimulation", "Simulation", "SpongeSimulation", "config"]
