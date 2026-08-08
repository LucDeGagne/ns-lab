"""Taylor-Green vortex exact solution utilities."""

from __future__ import annotations

import numpy as np

from ns_lab.grids import Grid2D


def _decay_factor(viscosity: float, time: float, wave_number: float = 1.0) -> float:
    """Return the Taylor-Green velocity/vorticity decay factor."""
    if viscosity < 0.0:
        msg = "viscosity must be nonnegative."
        raise ValueError(msg)

    if time < 0.0:
        msg = "time must be nonnegative."
        raise ValueError(msg)

    if wave_number <= 0.0:
        msg = "wave_number must be positive."
        raise ValueError(msg)

    return float(np.exp(-2.0 * viscosity * wave_number**2 * time))


def taylor_green_velocity(
    grid: Grid2D,
    viscosity: float,
    time: float,
    *,
    amplitude: float = 1.0,
    wave_number: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact 2D Taylor-Green velocity field.

    Uses:

        u = A sin(kx) cos(ky) exp(-2νk²t)
        v = -A cos(kx) sin(ky) exp(-2νk²t)
    """
    if amplitude <= 0.0:
        msg = "amplitude must be positive."
        raise ValueError(msg)

    x, y = grid.mesh
    decay = _decay_factor(viscosity, time, wave_number)

    u = amplitude * np.sin(wave_number * x) * np.cos(wave_number * y) * decay
    v = -amplitude * np.cos(wave_number * x) * np.sin(wave_number * y) * decay

    return u, v


def taylor_green_vorticity(
    grid: Grid2D,
    viscosity: float,
    time: float,
    *,
    amplitude: float = 1.0,
    wave_number: float = 1.0,
) -> np.ndarray:
    """Return the exact 2D Taylor-Green vorticity field.

    Uses:

        ω = 2Ak sin(kx) sin(ky) exp(-2νk²t)
    """
    if amplitude <= 0.0:
        msg = "amplitude must be positive."
        raise ValueError(msg)

    x, y = grid.mesh
    decay = _decay_factor(viscosity, time, wave_number)

    return (
        2.0
        * amplitude
        * wave_number
        * np.sin(wave_number * x)
        * np.sin(wave_number * y)
        * decay
    )


def taylor_green_energy(
    viscosity: float,
    time: float,
    *,
    amplitude: float = 1.0,
    wave_number: float = 1.0,
) -> float:
    """Return the exact kinetic energy on the default 2π by 2π domain.

    For amplitude A and wave number k = 1 on [0, 2π]²:

        E(t) = π² A² exp(-4νt)
    """
    if amplitude <= 0.0:
        msg = "amplitude must be positive."
        raise ValueError(msg)

    decay = _decay_factor(viscosity, time, wave_number)

    return float(np.pi**2 * amplitude**2 * decay**2)


def taylor_green_enstrophy(
    viscosity: float,
    time: float,
    *,
    amplitude: float = 1.0,
    wave_number: float = 1.0,
) -> float:
    """Return the exact enstrophy on the default 2π by 2π domain.

    For amplitude A and wave number k = 1 on [0, 2π]²:

        E_ω(t) = 2π² A² exp(-4νt)
    """
    if amplitude <= 0.0:
        msg = "amplitude must be positive."
        raise ValueError(msg)

    decay = _decay_factor(viscosity, time, wave_number)

    return float(2.0 * np.pi**2 * amplitude**2 * wave_number**2 * decay**2)