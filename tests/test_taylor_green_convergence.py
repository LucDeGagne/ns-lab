import numpy as np

from ns_lab.benchmarks import taylor_green_vorticity
from ns_lab.grids import Grid2D
from ns_lab.solvers.vorticity2d import euler_step


def _run_taylor_green_error(dt: float) -> float:
    grid = Grid2D(nx=64, ny=64)
    viscosity = 0.01
    final_time = 0.5
    steps = int(final_time / dt)

    omega = taylor_green_vorticity(grid, viscosity=viscosity, time=0.0)

    for _ in range(steps):
        omega = euler_step(omega, grid, viscosity=viscosity, dt=dt)

    exact_omega = taylor_green_vorticity(
        grid,
        viscosity=viscosity,
        time=final_time,
    )

    return float(np.max(np.abs(omega - exact_omega)))


def test_taylor_green_error_decreases_when_timestep_decreases() -> None:
    coarse_error = _run_taylor_green_error(dt=0.004)
    medium_error = _run_taylor_green_error(dt=0.002)
    fine_error = _run_taylor_green_error(dt=0.001)

    assert medium_error < coarse_error
    assert fine_error < medium_error


def test_taylor_green_euler_is_approximately_first_order_in_time() -> None:
    coarse_error = _run_taylor_green_error(dt=0.004)
    medium_error = _run_taylor_green_error(dt=0.002)
    fine_error = _run_taylor_green_error(dt=0.001)

    coarse_to_medium_ratio = coarse_error / medium_error
    medium_to_fine_ratio = medium_error / fine_error

    assert 1.8 < coarse_to_medium_ratio < 2.2
    assert 1.8 < medium_to_fine_ratio < 2.2