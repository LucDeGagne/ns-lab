"""Run the 2D Taylor-Green vortex benchmark.

This example starts from the exact Taylor-Green vorticity field at t = 0,
evolves it numerically with the current explicit Euler solver, and compares the
result against the exact solution at the final time.
"""

from __future__ import annotations

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from ns_lab.benchmarks import (
    taylor_green_energy,
    taylor_green_enstrophy,
    taylor_green_vorticity,
)
from ns_lab.diagnostics import energy, enstrophy
from ns_lab.grids import Grid2D
from ns_lab.plotting import plot_vorticity_difference
from ns_lab.solvers.vorticity2d import euler_step
from ns_lab.vorticity import velocity_from_vorticity


def main() -> None:
    grid = Grid2D(nx=64, ny=64)

    viscosity = 0.01
    dt = 0.001
    steps = 500
    final_time = steps * dt

    omega = taylor_green_vorticity(grid, viscosity=viscosity, time=0.0)
    initial_omega = omega.copy()

    start = perf_counter()

    for _ in range(steps):
        omega = euler_step(omega, grid, viscosity=viscosity, dt=dt)

    elapsed = perf_counter() - start

    exact_final_omega = taylor_green_vorticity(
        grid,
        viscosity=viscosity,
        time=final_time,
    )

    u_final, v_final = velocity_from_vorticity(omega, grid)

    numerical_energy = energy(u_final, v_final, grid)
    exact_energy = taylor_green_energy(viscosity, final_time)

    numerical_enstrophy = enstrophy(omega, grid)
    exact_enstrophy = taylor_green_enstrophy(viscosity, final_time)

    max_vorticity_error = float(np.max(np.abs(omega - exact_final_omega)))

    print("Run summary")
    print(f"  Grid:             {grid.nx} x {grid.ny}")
    print(f"  Steps:            {steps}")
    print(f"  Simulated time:   {final_time:.6f}")
    print(f"  Runtime:          {elapsed:.3f} seconds")
    print(f"  Steps per second: {steps / elapsed:.1f}")
    print()

    print("Final diagnostics")
    print(f"  Numerical energy:      {numerical_energy:.8f}")
    print(f"  Exact energy:          {exact_energy:.8f}")
    print(f"  Energy error:          {abs(numerical_energy - exact_energy):.3e}")
    print(f"  Numerical enstrophy:   {numerical_enstrophy:.8f}")
    print(f"  Exact enstrophy:       {exact_enstrophy:.8f}")
    print(f"  Enstrophy error:       {abs(numerical_enstrophy - exact_enstrophy):.3e}")
    print(f"  Max vorticity error:   {max_vorticity_error:.3e}")

    plot_vorticity_difference(
        initial_omega,
        omega,
        grid,
        initial_title="Taylor-Green initial vorticity",
        final_title="Taylor-Green numerical final vorticity",
        difference_title="Numerical final - initial",
    )
    plt.show()


if __name__ == "__main__":
    main()