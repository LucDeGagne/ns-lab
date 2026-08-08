import numpy as np
import pytest

from ns_lab.benchmarks import (
    taylor_green_energy,
    taylor_green_enstrophy,
    taylor_green_velocity,
    taylor_green_vorticity,
)
from ns_lab.diagnostics import energy, enstrophy
from ns_lab.grids import Grid2D
from ns_lab.vorticity import velocity_from_vorticity


def test_taylor_green_velocity_shapes() -> None:
    grid = Grid2D(nx=16, ny=16)

    u, v = taylor_green_velocity(grid, viscosity=0.01, time=0.5)

    assert u.shape == grid.shape
    assert v.shape == grid.shape


def test_taylor_green_vorticity_shape() -> None:
    grid = Grid2D(nx=16, ny=16)

    omega = taylor_green_vorticity(grid, viscosity=0.01, time=0.5)

    assert omega.shape == grid.shape


def test_taylor_green_vorticity_matches_velocity_curl() -> None:
    grid = Grid2D(nx=32, ny=32)
    viscosity = 0.01
    time = 0.5

    omega = taylor_green_vorticity(grid, viscosity=viscosity, time=time)
    u, v = velocity_from_vorticity(omega, grid)
    expected_u, expected_v = taylor_green_velocity(grid, viscosity=viscosity, time=time)

    np.testing.assert_allclose(u, expected_u, atol=1e-12)
    np.testing.assert_allclose(v, expected_v, atol=1e-12)


def test_taylor_green_energy_matches_diagnostic() -> None:
    grid = Grid2D(nx=64, ny=64)
    viscosity = 0.01
    time = 0.5

    u, v = taylor_green_velocity(grid, viscosity=viscosity, time=time)

    assert energy(u, v, grid) == pytest.approx(
        taylor_green_energy(viscosity, time),
        rel=1e-12,
    )


def test_taylor_green_enstrophy_matches_diagnostic() -> None:
    grid = Grid2D(nx=64, ny=64)
    viscosity = 0.01
    time = 0.5

    omega = taylor_green_vorticity(grid, viscosity=viscosity, time=time)

    assert enstrophy(omega, grid) == pytest.approx(
        taylor_green_enstrophy(viscosity, time),
        rel=1e-12,
    )


def test_taylor_green_exact_values_at_time_zero() -> None:
    viscosity = 0.01
    time = 0.0

    assert taylor_green_energy(viscosity, time) == pytest.approx(np.pi**2)
    assert taylor_green_enstrophy(viscosity, time) == pytest.approx(2.0 * np.pi**2)


def test_taylor_green_rejects_invalid_inputs() -> None:
    grid = Grid2D(nx=8, ny=8)

    with pytest.raises(ValueError, match="viscosity must be nonnegative"):
        taylor_green_velocity(grid, viscosity=-0.01, time=0.0)

    with pytest.raises(ValueError, match="time must be nonnegative"):
        taylor_green_vorticity(grid, viscosity=0.01, time=-1.0)

    with pytest.raises(ValueError, match="amplitude must be positive"):
        taylor_green_velocity(grid, viscosity=0.01, time=0.0, amplitude=0.0)

    with pytest.raises(ValueError, match="wave_number must be positive"):
        taylor_green_vorticity(grid, viscosity=0.01, time=0.0, wave_number=0.0)