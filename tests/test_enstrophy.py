import numpy as np
import pytest

from ns_lab.diagnostics import enstrophy
from ns_lab.grids import Grid2D


def test_enstrophy_of_zero_field_is_zero() -> None:
    grid = Grid2D(nx=8, ny=8)
    omega = np.zeros(grid.shape)

    assert enstrophy(omega, grid) == 0.0


def test_enstrophy_of_constant_field_matches_area_integral() -> None:
    grid = Grid2D(nx=8, ny=8, length_x=2.0, length_y=3.0)
    omega = 2.0 * np.ones(grid.shape)

    expected = 0.5 * 2.0**2 * grid.length_x * grid.length_y

    assert enstrophy(omega, grid) == pytest.approx(expected)


def test_enstrophy_of_sine_wave_on_default_domain() -> None:
    grid = Grid2D(nx=64, ny=64)
    x, _ = grid.mesh

    omega = np.sin(x)

    expected = np.pi**2

    assert enstrophy(omega, grid) == pytest.approx(expected)


def test_enstrophy_rejects_wrong_shape() -> None:
    grid = Grid2D(nx=8, ny=8)
    omega = np.zeros((4, 4))

    with pytest.raises(ValueError, match="omega shape must be"):
        enstrophy(omega, grid)