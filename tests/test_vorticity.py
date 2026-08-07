import numpy as np
import pytest

from ns_lab.grids import Grid2D
from ns_lab.spectral import derivative_x, derivative_y
from ns_lab.vorticity import velocity_from_vorticity


def test_velocity_from_vorticity_matches_known_streamfunction() -> None:
    grid = Grid2D(nx=32, ny=32)
    x, y = grid.mesh

    omega = 2.0 * np.sin(x) * np.cos(y)

    expected_u = -np.sin(x) * np.sin(y)
    expected_v = -np.cos(x) * np.cos(y)

    u, v = velocity_from_vorticity(omega, grid)

    np.testing.assert_allclose(u, expected_u, atol=1e-12)
    np.testing.assert_allclose(v, expected_v, atol=1e-12)


def test_velocity_from_vorticity_is_divergence_free() -> None:
    grid = Grid2D(nx=32, ny=32)
    x, y = grid.mesh

    omega = np.sin(x) + np.cos(2.0 * y)

    u, v = velocity_from_vorticity(omega, grid)

    divergence = derivative_x(u, grid) + derivative_y(v, grid)

    np.testing.assert_allclose(divergence, np.zeros(grid.shape), atol=1e-12)


def test_velocity_from_vorticity_reconstructs_original_vorticity() -> None:
    grid = Grid2D(nx=32, ny=32)
    x, y = grid.mesh

    omega = np.sin(x) + np.cos(2.0 * y)

    u, v = velocity_from_vorticity(omega, grid)

    reconstructed_omega = derivative_x(v, grid) - derivative_y(u, grid)

    np.testing.assert_allclose(reconstructed_omega, omega, atol=1e-12)


def test_velocity_from_vorticity_rejects_wrong_shape() -> None:
    grid = Grid2D(nx=32, ny=32)
    omega = np.zeros((16, 16))

    with pytest.raises(ValueError, match="field shape must be"):
        velocity_from_vorticity(omega, grid)


def test_velocity_from_vorticity_rejects_nonzero_mean() -> None:
    grid = Grid2D(nx=32, ny=32)
    omega = np.ones(grid.shape)

    with pytest.raises(ValueError, match="mean vorticity must be near zero"):
        velocity_from_vorticity(omega, grid)