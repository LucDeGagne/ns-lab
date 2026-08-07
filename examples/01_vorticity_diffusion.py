"""Run a simple 2D vorticity diffusion demo.

This example starts with a single smooth sine-wave vorticity field and evolves it
forward with viscosity. For this setup, advection is effectively zero, so the
vorticity should slowly decay rather than swirl into complicated structures.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from ns_lab.diagnostics import energy, enstrophy
from ns_lab.grids import Grid2D
from ns_lab.plotting import plot_vorticity_comparison
from ns_lab.solvers.vorticity2d import euler_step
from ns_lab.vorticity import velocity_from_vorticity


def main() -> None:
    grid = Grid2D(nx=64, ny=64)

    viscosity = 0.01
    dt = 0.001
    steps = 500

    x, _ = grid.mesh
    omega = np.sin(x)
    initial_omega = omega.copy()

    start = perf_counter()

    for _ in range(steps):
        omega = euler_step(omega, grid, viscosity=viscosity, dt=dt)

    elapsed = perf_counter() - start

    u_initial, v_initial = velocity_from_vorticity(initial_omega, grid)
    u_final, v_final = velocity_from_vorticity(omega, grid)

    print("Run summary")
    print(f"  Steps:            {steps}")
    print(f"  Simulated time:   {steps * dt:.6f}")
    print(f"  Runtime:          {elapsed:.3f} seconds")
    print(f"  Steps per second: {steps / elapsed:.1f}")
    print()

    print("Initial diagnostics")
    print(f"  Energy:    {energy(u_initial, v_initial, grid):.8f}")
    print(f"  Enstrophy: {enstrophy(initial_omega, grid):.8f}")

    print("Final diagnostics")
    print(f"  Energy:    {energy(u_final, v_final, grid):.8f}")
    print(f"  Enstrophy: {enstrophy(omega, grid):.8f}")

    plot_vorticity_comparison(initial_omega, omega, grid)
    plt.show()


if __name__ == "__main__":
    main()