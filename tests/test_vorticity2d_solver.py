import numpy as np
import pytest

from ns_lab.grids import Grid2D
from ns_lab.solvers.vorticity2d import vorticity_rhs


def test_vorticity_rhs_of_zero_field_is_zero() -> None:
    grid = Grid2D(nx=32, ny=32)
    omega = np.zeros(grid.shape)

    rhs = vorticity_rhs(omega, grid, viscosity=0.001)

    np.testing.assert_allclose(rhs, np.zeros(grid.shape), atol=1e-12)


def test_vorticity_rhs_diffuses_single_sine_mode() -> None:
    grid = Grid2D(nx=32, ny=32)
    x, _ = grid.mesh

    viscosity = 0.001
    omega = np.sin(x)

    rhs = vorticity_rhs(omega, grid, viscosity=viscosity)

    expected = -viscosity * np.sin(x)

    np.testing.assert_allclose(rhs, expected, atol=1e-12)


def test_vorticity_rhs_rejects_negative_viscosity() -> None:
    grid = Grid2D(nx=32, ny=32)
    omega = np.zeros(grid.shape)

    with pytest.raises(ValueError, match="viscosity must be nonnegative"):
        vorticity_rhs(omega, grid, viscosity=-0.001)