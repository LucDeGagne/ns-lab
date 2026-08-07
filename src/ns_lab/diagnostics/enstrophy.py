"""Enstrophy diagnostics for vorticity fields."""

from __future__ import annotations

import numpy as np

from ns_lab.grids import Grid2D


def enstrophy(omega: np.ndarray, grid: Grid2D) -> float:
    """Compute the enstrophy of a 2D vorticity field.

    Uses:

        E_ω = 1/2 ∫ ω² dA

    On a uniform grid, this is approximated by:

        E_ω ≈ 1/2 * sum(ωᵢⱼ²) * dx * dy
    """
    if omega.shape != grid.shape:
        msg = f"omega shape must be {grid.shape}, got {omega.shape}."
        raise ValueError(msg)

    return 0.5 * float(np.sum(omega**2) * grid.dx * grid.dy)