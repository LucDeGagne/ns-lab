"""Benchmark flows and exact solutions for NS Lab."""

from ns_lab.benchmarks.taylor_green import (
    taylor_green_energy,
    taylor_green_enstrophy,
    taylor_green_velocity,
    taylor_green_vorticity,
)

__all__ = [
    "taylor_green_energy",
    "taylor_green_enstrophy",
    "taylor_green_velocity",
    "taylor_green_vorticity",
]