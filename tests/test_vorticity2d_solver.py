import numpy as np
import pytest

from ns_lab.grids import Grid2D
from ns_lab.solvers.vorticity2d import euler_step, vorticity_rhs


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


def test_euler_step_of_zero_field_stays_zero() -> None:
    grid = Grid2D(nx=32, ny=32)
    omega = np.zeros(grid.shape)

    omega_next = euler_step(omega, grid, viscosity=0.001, dt=0.01)

    np.testing.assert_allclose(omega_next, np.zeros(grid.shape), atol=1e-12)


def test_euler_step_diffuses_single_sine_mode() -> None:
    grid = Grid2D(nx=32, ny=32)
    x, _ = grid.mesh

    viscosity = 0.001
    dt = 0.01
    omega = np.sin(x)

    omega_next = euler_step(omega, grid, viscosity=viscosity, dt=dt)

    expected = omega + dt * (-viscosity * np.sin(x))

    np.testing.assert_allclose(omega_next, expected, atol=1e-12)


def test_euler_step_rejects_nonpositive_dt() -> None:
    grid = Grid2D(nx=32, ny=32)
    omega = np.zeros(grid.shape)

    with pytest.raises(ValueError, match="dt must be positive"):
        euler_step(omega, grid, viscosity=0.001, dt=0.0)

    with pytest.raises(ValueError, match="dt must be positive"):
        euler_step(omega, grid, viscosity=0.001, dt=-0.01)