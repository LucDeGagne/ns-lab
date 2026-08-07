import numpy as np
import pytest

from ns_lab.diagnostics import energy
from ns_lab.grids import Grid2D


def test_energy_of_zero_velocity_is_zero() -> None:
    grid = Grid2D(nx=8, ny=8)
    u = np.zeros(grid.shape)
    v = np.zeros(grid.shape)

    assert energy(u, v, grid) == 0.0


def test_energy_of_constant_velocity_matches_area_integral() -> None:
    grid = Grid2D(nx=8, ny=8, length_x=2.0, length_y=3.0)
    u = 2.0 * np.ones(grid.shape)
    v = 1.0 * np.ones(grid.shape)

    expected = 0.5 * (2.0**2 + 1.0**2) * grid.length_x * grid.length_y

    assert energy(u, v, grid) == pytest.approx(expected)


def test_energy_of_sine_velocity_on_default_domain() -> None:
    grid = Grid2D(nx=64, ny=64)
    x, _ = grid.mesh

    u = np.sin(x)
    v = np.zeros(grid.shape)

    expected = np.pi**2

    assert energy(u, v, grid) == pytest.approx(expected)


def test_energy_rejects_wrong_shapes() -> None:
    grid = Grid2D(nx=8, ny=8)
    valid = np.zeros(grid.shape)
    invalid = np.zeros((4, 4))

    with pytest.raises(ValueError, match="u shape must be"):
        energy(invalid, valid, grid)

    with pytest.raises(ValueError, match="v shape must be"):
        energy(valid, invalid, grid)