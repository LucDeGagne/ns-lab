"""Plotting helpers for vorticity fields."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ns_lab.grids import Grid2D


def _validate_vorticity_field(omega: np.ndarray, grid: Grid2D, name: str) -> None:
    """Validate that a vorticity field matches the grid shape."""
    if omega.shape != grid.shape:
        msg = f"{name} shape must be {grid.shape}, got {omega.shape}."
        raise ValueError(msg)


def plot_vorticity_comparison(
    initial_omega: np.ndarray,
    final_omega: np.ndarray,
    grid: Grid2D,
    *,
    initial_title: str = "Initial vorticity",
    final_title: str = "Final vorticity",
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot initial and final vorticity fields side by side."""
    _validate_vorticity_field(initial_omega, grid, "initial_omega")
    _validate_vorticity_field(final_omega, grid, "final_omega")

    max_abs = max(
        float(np.max(np.abs(initial_omega))),
        float(np.max(np.abs(final_omega))),
    )

    if max_abs == 0.0:
        max_abs = 1.0

    vmin = -max_abs
    vmax = max_abs

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    initial_plot = axes[0].imshow(
        initial_omega.T,
        origin="lower",
        extent=(0.0, grid.length_x, 0.0, grid.length_y),
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(initial_title)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(initial_plot, ax=axes[0])

    final_plot = axes[1].imshow(
        final_omega.T,
        origin="lower",
        extent=(0.0, grid.length_x, 0.0, grid.length_y),
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(final_title)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(final_plot, ax=axes[1])

    return fig, (axes[0], axes[1])