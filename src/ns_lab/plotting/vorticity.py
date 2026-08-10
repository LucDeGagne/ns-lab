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


def _symmetric_color_limits(*fields: np.ndarray) -> tuple[float, float]:
    """Return symmetric color limits around zero for one or more fields."""
    max_abs = max(float(np.max(np.abs(field))) for field in fields)

    if max_abs == 0.0:
        max_abs = 1.0

    return -max_abs, max_abs


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

    vmin, vmax = _symmetric_color_limits(initial_omega, final_omega)

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


def plot_vorticity_difference(
    initial_omega: np.ndarray,
    final_omega: np.ndarray,
    grid: Grid2D,
    *,
    initial_title: str = "Initial vorticity",
    final_title: str = "Final vorticity",
    difference_title: str = "Difference",
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """Plot initial, final, and difference vorticity fields.

    The difference panel shows:

        final_omega - initial_omega
    """
    _validate_vorticity_field(initial_omega, grid, "initial_omega")
    _validate_vorticity_field(final_omega, grid, "final_omega")

    difference = final_omega - initial_omega

    vmin, vmax = _symmetric_color_limits(initial_omega, final_omega)
    diff_vmin, diff_vmax = _symmetric_color_limits(difference)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

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

    difference_plot = axes[2].imshow(
        difference.T,
        origin="lower",
        extent=(0.0, grid.length_x, 0.0, grid.length_y),
        vmin=diff_vmin,
        vmax=diff_vmax,
    )
    axes[2].set_title(difference_title)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(difference_plot, ax=axes[2])

    return fig, (axes[0], axes[1], axes[2])

def plot_vorticity_error(
    numerical_omega: np.ndarray,
    exact_omega: np.ndarray,
    grid: Grid2D,
    *,
    numerical_title: str = "Numerical vorticity",
    exact_title: str = "Exact vorticity",
    error_title: str = "Numerical - exact",
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """Plot numerical, exact, and error vorticity fields.

    The error panel shows:

        numerical_omega - exact_omega
    """
    _validate_vorticity_field(numerical_omega, grid, "numerical_omega")
    _validate_vorticity_field(exact_omega, grid, "exact_omega")

    error = numerical_omega - exact_omega

    vmin, vmax = _symmetric_color_limits(numerical_omega, exact_omega)
    error_vmin, error_vmax = _symmetric_color_limits(error)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    numerical_plot = axes[0].imshow(
        numerical_omega.T,
        origin="lower",
        extent=(0.0, grid.length_x, 0.0, grid.length_y),
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(numerical_title)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(numerical_plot, ax=axes[0])

    exact_plot = axes[1].imshow(
        exact_omega.T,
        origin="lower",
        extent=(0.0, grid.length_x, 0.0, grid.length_y),
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(exact_title)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(exact_plot, ax=axes[1])

    error_plot = axes[2].imshow(
        error.T,
        origin="lower",
        extent=(0.0, grid.length_x, 0.0, grid.length_y),
        vmin=error_vmin,
        vmax=error_vmax,
    )
    axes[2].set_title(error_title)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(error_plot, ax=axes[2])

    return fig, (axes[0], axes[1], axes[2])