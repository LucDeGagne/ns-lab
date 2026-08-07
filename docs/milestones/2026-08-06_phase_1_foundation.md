# Phase 1 Foundation

Date: 2026-08-06

## Summary

NS Lab now has a working, tested foundation for simple 2D periodic vorticity simulations.

## Built

- Python package structure using `src/ns_lab`
- Uniform periodic 2D grid
- Spectral derivative utilities
- Spectral Laplacian
- Velocity recovery from vorticity
- 2D vorticity right-hand side
- Explicit Euler timestepper
- Energy diagnostic
- Enstrophy diagnostic
- Reusable vorticity plotting helper
- First runnable vorticity diffusion example

## Validation

- Test suite passing with 34 tests
- First example successfully runs a simple viscous vorticity diffusion case
- Energy and enstrophy decrease as expected
- Runtime baseline recorded at roughly 500 steps in 0.464 seconds on local laptop

## Notes

The first example is intentionally simple. It is not yet a serious benchmark or research reproduction. It demonstrates that the package can run, measure, and plot a basic 2D vorticity simulation.

## Next Phase

Phase 2 begins with paper-backed 2D benchmark reproduction.

First target:

- Taylor-Green vortex
- Exact solution helpers
- Expected energy and enstrophy values
- Numerical comparison against known analytic values
- Example script
- Research note