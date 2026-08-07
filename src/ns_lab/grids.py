"""Grid definitions for NS Lab simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    """A uniform 2D periodic grid.

    Parameters
    ----------
    nx:
        Number of grid points in the x direction.
    ny:
        Number of grid points in the y direction.
    length_x:
        Physical length of the domain in the x direction.
    length_y:
        Physical length of the domain in the y direction.
    """

    nx: int
    ny: int
    length_x: float = 2.0 * np.pi
    length_y: float = 2.0 * np.pi

    def __post_init__(self) -> None:
        if self.nx <= 0:
            msg = "nx must be positive."
            raise ValueError(msg)

        if self.ny <= 0:
            msg = "ny must be positive."
            raise ValueError(msg)

        if self.length_x <= 0:
            msg = "length_x must be positive."
            raise ValueError(msg)

        if self.length_y <= 0:
            msg = "length_y must be positive."
            raise ValueError(msg)

    @property
    def dx(self) -> float:
        """Grid spacing in the x direction."""
        return self.length_x / self.nx

    @property
    def dy(self) -> float:
        """Grid spacing in the y direction."""
        return self.length_y / self.ny

    @property
    def x(self) -> np.ndarray:
        """One-dimensional x coordinates, excluding the periodic endpoint."""
        return np.linspace(0.0, self.length_x, self.nx, endpoint=False)

    @property
    def y(self) -> np.ndarray:
        """One-dimensional y coordinates, excluding the periodic endpoint."""
        return np.linspace(0.0, self.length_y, self.ny, endpoint=False)

    @property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Two-dimensional coordinate arrays."""
        return np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def shape(self) -> tuple[int, int]:
        """Grid array shape."""
        return self.nx, self.ny