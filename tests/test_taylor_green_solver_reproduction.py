import numpy as np

from ns_lab.benchmarks import (
    taylor_green_energy,
    taylor_green_enstrophy,
    taylor_green_vorticity,
)
from ns_lab.diagnostics import energy, enstrophy
from ns_lab.grids import Grid2D
from ns_lab.solvers.vorticity2d import euler_step
from ns_lab.vorticity import velocity_from_vorticity


def test_taylor_green_solver_reproduces_exact_decay() -> None:
    grid = Grid2D(nx=64, ny=64)
    viscosity = 0.01
    dt = 0.001
    steps = 500
    final_time = steps * dt

    omega = taylor_green_vorticity(grid, viscosity=viscosity, time=0.0)

    for _ in range(steps):
        omega = euler_step(omega, grid, viscosity=viscosity, dt=dt)

    exact_omega = taylor_green_vorticity(
        grid,
        viscosity=viscosity,
        time=final_time,
    )

    u, v = velocity_from_vorticity(omega, grid)

    assert abs(energy(u, v, grid) - taylor_green_energy(viscosity, final_time)) < 2.5e-6
    assert (
        abs(enstrophy(omega, grid) - taylor_green_enstrophy(viscosity, final_time))
        < 5.0e-6
    )
    assert float(np.max(np.abs(omega - exact_omega))) < 2.5e-7