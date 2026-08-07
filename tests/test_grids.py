import numpy as np
import pytest

from ns_lab.grids import Grid2D


def test_grid2d_has_expected_spacing() -> None:
    grid = Grid2D(nx=4, ny=8, length_x=2.0, length_y=4.0)

    assert grid.dx == 0.5
    assert grid.dy == 0.5


def test_grid2d_coordinates_exclude_periodic_endpoint() -> None:
    grid = Grid2D(nx=4, ny=4, length_x=2.0, length_y=2.0)

    np.testing.assert_allclose(grid.x, np.array([0.0, 0.5, 1.0, 1.5]))
    np.testing.assert_allclose(grid.y, np.array([0.0, 0.5, 1.0, 1.5]))


def test_grid2d_mesh_has_expected_shape() -> None:
    grid = Grid2D(nx=4, ny=8)

    x, y = grid.mesh

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_grid2d_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="nx must be positive"):
        Grid2D(nx=0, ny=8)

    with pytest.raises(ValueError, match="ny must be positive"):
        Grid2D(nx=8, ny=0)


def test_grid2d_wave_numbers_for_default_periodic_domain() -> None:
    grid = Grid2D(nx=4, ny=4)

    np.testing.assert_allclose(grid.wave_numbers_x, np.array([0.0, 1.0, -2.0, -1.0]))
    np.testing.assert_allclose(grid.wave_numbers_y, np.array([0.0, 1.0, -2.0, -1.0]))


def test_grid2d_wave_number_mesh_has_expected_shape() -> None:
    grid = Grid2D(nx=4, ny=8)

    kx, ky = grid.wave_number_mesh

    assert kx.shape == (4, 8)
    assert ky.shape == (4, 8)


def test_grid2d_wave_number_squared_is_nonnegative() -> None:
    grid = Grid2D(nx=4, ny=4)

    assert grid.wave_number_squared.shape == (4, 4)
    assert np.all(grid.wave_number_squared >= 0.0)
    assert grid.wave_number_squared[0, 0] == 0.0

