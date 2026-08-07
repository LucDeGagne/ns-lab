"""Vorticity utilities for incompressible 2D flow."""

from __future__ import annotations

import numpy as np

from ns_lab.grids import Grid2D


def _validate_scalar_field(field: np.ndarray, grid: Grid2D) -> None:
    """Validate that a scalar field matches the grid shape."""
    if field.shape != grid.shape:
        msg = f"field shape must be {grid.shape}, got {field.shape}."
        raise ValueError(msg)


def velocity_from_vorticity(
    omega: np.ndarray,
    grid: Grid2D,
    *,
    mean_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover incompressible 2D velocity from scalar vorticity.

    Uses the streamfunction convention:

        u = dψ/dy
        v = -dψ/dx
        ω = -Δψ

    In Fourier space:

        ψ_hat = ω_hat / k^2

    The zero mode is set to zero. A nonzero mean vorticity cannot be represented
    by a periodic velocity field under this convention.
    """
    _validate_scalar_field(omega, grid)

    mean_vorticity = float(np.mean(omega))
    if abs(mean_vorticity) > mean_tolerance:
        msg = f"mean vorticity must be near zero, got {mean_vorticity}."
        raise ValueError(msg)

    omega_hat = np.fft.fft2(omega)

    kx, ky = grid.wave_number_mesh
    k_squared = grid.wave_number_squared

    psi_hat = np.zeros_like(omega_hat, dtype=np.complex128)
    nonzero = k_squared > 0.0
    psi_hat[nonzero] = omega_hat[nonzero] / k_squared[nonzero]

    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat

    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real

    return u, v