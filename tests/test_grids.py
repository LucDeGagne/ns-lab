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