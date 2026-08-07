"""Energy diagnostics for velocity fields."""

from __future__ import annotations

import numpy as np

from ns_lab.grids import Grid2D


def energy(u: np.ndarray, v: np.ndarray, grid: Grid2D) -> float:
    """Compute the kinetic energy of a 2D velocity field.

    Uses:

        E = 1/2 ∫ (u² + v²) dA

    On a uniform grid, this is approximated by:

        E ≈ 1/2 * sum(uᵢⱼ² + vᵢⱼ²) * dx * dy
    """
    if u.shape != grid.shape:
        msg = f"u shape must be {grid.shape}, got {u.shape}."
        raise ValueError(msg)

    if v.shape != grid.shape:
        msg = f"v shape must be {grid.shape}, got {v.shape}."
        raise ValueError(msg)

    return 0.5 * float(np.sum(u**2 + v**2) * grid.dx * grid.dy)