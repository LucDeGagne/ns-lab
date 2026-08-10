import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ns_lab.grids import Grid2D
from ns_lab.plotting import plot_vorticity_comparison


def test_plot_vorticity_comparison_returns_figure_and_axes() -> None:
    grid = Grid2D(nx=8, ny=8)
    initial_omega = np.zeros(grid.shape)
    final_omega = np.ones(grid.shape)

    fig, axes = plot_vorticity_comparison(initial_omega, final_omega, grid)

    assert fig is not None
    assert len(axes) == 2
    assert axes[0].get_title() == "Initial vorticity"
    assert axes[1].get_title() == "Final vorticity"


def test_plot_vorticity_comparison_rejects_wrong_shapes() -> None:
    grid = Grid2D(nx=8, ny=8)
    valid = np.zeros(grid.shape)
    invalid = np.zeros((4, 4))

    with pytest.raises(ValueError, match="initial_omega shape must be"):
        plot_vorticity_comparison(invalid, valid, grid)

    with pytest.raises(ValueError, match="final_omega shape must be"):
        plot_vorticity_comparison(valid, invalid, grid)


def test_plot_vorticity_difference_returns_figure_and_three_axes() -> None:
    grid = Grid2D(nx=8, ny=8)
    initial_omega = np.zeros(grid.shape)
    final_omega = np.ones(grid.shape)

    from ns_lab.plotting import plot_vorticity_difference

    fig, axes = plot_vorticity_difference(initial_omega, final_omega, grid)

    assert fig is not None
    assert len(axes) == 3
    assert axes[0].get_title() == "Initial vorticity"
    assert axes[1].get_title() == "Final vorticity"
    assert axes[2].get_title() == "Difference"


def test_plot_vorticity_difference_rejects_wrong_shapes() -> None:
    grid = Grid2D(nx=8, ny=8)
    valid = np.zeros(grid.shape)
    invalid = np.zeros((4, 4))

    from ns_lab.plotting import plot_vorticity_difference

    with pytest.raises(ValueError, match="initial_omega shape must be"):
        plot_vorticity_difference(invalid, valid, grid)

    with pytest.raises(ValueError, match="final_omega shape must be"):
        plot_vorticity_difference(valid, invalid, grid)