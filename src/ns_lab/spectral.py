"""Spectral operations for periodic fields."""

from __future__ import annotations

import numpy as np

from ns_lab.grids import Grid2D


def _validate_scalar_field(field: np.ndarray, grid: Grid2D) -> None:
    """Validate that a scalar field matches the grid shape."""
    if field.shape != grid.shape:
        msg = f"field shape must be {grid.shape}, got {field.shape}."
        raise ValueError(msg)


def derivative_x(field: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Compute the x derivative of a scalar periodic field.

    Uses the Fourier-space identity:

        d/dx f  <-->  i * kx * f_hat
    """
    _validate_scalar_field(field, grid)

    field_hat = np.fft.fft2(field)
    kx, _ = grid.wave_number_mesh
    derivative_hat = 1j * kx * field_hat

    return np.fft.ifft2(derivative_hat).real


def derivative_y(field: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Compute the y derivative of a scalar periodic field.

    Uses the Fourier-space identity:

        d/dy f  <-->  i * ky * f_hat
    """
    _validate_scalar_field(field, grid)

    field_hat = np.fft.fft2(field)
    _, ky = grid.wave_number_mesh
    derivative_hat = 1j * ky * field_hat

    return np.fft.ifft2(derivative_hat).real