"""2D vorticity-form Navier-Stokes utilities."""

from __future__ import annotations

import numpy as np

from ns_lab.grids import Grid2D
from ns_lab.spectral import derivative_x, derivative_y, laplacian
from ns_lab.vorticity import velocity_from_vorticity


def vorticity_rhs(omega: np.ndarray, grid: Grid2D, viscosity: float) -> np.ndarray:
    """Compute the right-hand side of the 2D vorticity equation.

    Uses:

        dω/dt = -(u dω/dx + v dω/dy) + ν Δω

    where velocity (u, v) is recovered from vorticity.
    """
    if viscosity < 0.0:
        msg = "viscosity must be nonnegative."
        raise ValueError(msg)

    u, v = velocity_from_vorticity(omega, grid)

    omega_x = derivative_x(omega, grid)
    omega_y = derivative_y(omega, grid)

    advection = u * omega_x + v * omega_y
    diffusion = viscosity * laplacian(omega, grid)

    return -advection + diffusion