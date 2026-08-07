import numpy as np
import pytest

from ns_lab.grids import Grid2D
from ns_lab.spectral import derivative_x, derivative_y


def test_derivative_x_of_sine_wave() -> None:
    grid = Grid2D(nx=32, ny=32)
    x, y = grid.mesh

    field = np.sin(x) + np.cos(2.0 * y)

    np.testing.assert_allclose(derivative_x(field, grid), np.cos(x), atol=1e-12)


def test_derivative_y_of_cosine_wave() -> None:
    grid = Grid2D(nx=32, ny=32)
    x, y = grid.mesh

    field = np.sin(x) + np.cos(2.0 * y)

    np.testing.assert_allclose(derivative_y(field, grid), -2.0 * np.sin(2.0 * y), atol=1e-12)


def test_derivatives_reject_wrong_shape() -> None:
    grid = Grid2D(nx=32, ny=32)
    field = np.zeros((16, 16))

    with pytest.raises(ValueError, match="field shape must be"):
        derivative_x(field, grid)

    with pytest.raises(ValueError, match="field shape must be"):
        derivative_y(field, grid)