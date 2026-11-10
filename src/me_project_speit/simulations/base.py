"""Base class of the simulations, with the EasyFEA steps they share.

Both simulations solve the same transient heat equation with explicit
convection. They differ by the mesh, the heat source and the time step, which
each subclass defines. Needs ``EasyFEA`` (imported lazily, so that importing the
package does not require it).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from . import config

KELVIN = 273.15  # temperatures are solved in K and returned in deg C


class Simulation(ABC):
    """A transient thermal simulation of the unit cube.

    Subclasses declare the parameters they accept (``param_defaults``,
    ``param_ranges``) and implement :meth:`run`. Simulations hold no state:
    everything is a class or static method, so they can be called from worker
    processes.
    """

    #: name -> default value of each parameter the simulation accepts.
    param_defaults: ClassVar[dict[str, float]]
    #: name -> (min, max) range used to sample each parameter.
    param_ranges: ClassVar[dict[str, tuple[float, float]]]

    @classmethod
    @abstractmethod
    def run(cls, *args, **params):
        """Run one simulation. See the subclasses for the arguments and result."""

    @classmethod
    def resolve_params(cls, params: dict) -> dict:
        """``params`` completed with the defaults; unknown names raise ``ValueError``."""
        unknown = set(params) - set(cls.param_defaults)
        if unknown:
            raise ValueError(
                f"Unknown simulation parameter(s) {sorted(unknown)}; "
                f"available: {sorted(cls.param_defaults)}"
            )
        return {**cls.param_defaults, **params}

    # --- EasyFEA steps shared by the subclasses -------------------------------

    @staticmethod
    def _init_thermal(mesh, t0: float, dt: float):
        """Thermal simulation on ``mesh`` at uniform temperature ``t0`` (K), time step ``dt``."""
        from EasyFEA import Models, Simulations

        model = Models.Thermal(k=config.CONDUCTIVITY, c=config.CP, thickness=1)
        simu = Simulations.Thermal(mesh, model)
        simu.rho = config.RHO
        simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)
        simu._Set_solutions(simu.problemType, np.ones(mesh.Nn) * t0)
        return simu

    @staticmethod
    def _convection(simu, nodes: np.ndarray, h: float) -> np.ndarray:
        """Convective flux (W.m-2) on ``nodes`` for the current temperature (explicit)."""
        return h * (config.T_EXTERIOR - simu.thermal[nodes])

    @staticmethod
    def _step(simu, *loads) -> None:
        """Advance one time step under the surface loads ``(nodes, flux)``."""
        simu.Bc_Init()
        for nodes, flux in loads:
            simu.add_surfLoad(nodes, [flux], ["t"])
        simu.Solve()
        simu.Save_Iter()

    @staticmethod
    def _celsius(simu) -> np.ndarray:
        """Current nodal temperature, in deg C."""
        return simu.thermal - KELVIN
