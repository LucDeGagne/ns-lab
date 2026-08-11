# Benchmark 001: Taylor-Green Vortex

## Summary

This benchmark reproduces the 2D Taylor-Green vortex, an exact incompressible Navier-Stokes solution on a periodic domain.

The goal is to verify that NS Lab can:

- represent the exact velocity and vorticity fields
- evolve the vorticity numerically
- compare the numerical result against the exact solution
- track energy and enstrophy decay
- verify timestep convergence for the current explicit Euler solver

## Domain

The benchmark uses the default periodic domain:

```text
[0, 2π] x [0, 2π]
```

## Exact velocity field

NS Lab uses:

```text
u(x, y, t) =  sin(x) cos(y) exp(-2νt)
v(x, y, t) = -cos(x) sin(y) exp(-2νt)
```

Plain English:

The velocity forms a repeating pattern of rotating cells. Viscosity causes the whole pattern to decay smoothly over time.

## Exact vorticity field

```text
ω(x, y, t) = 2 sin(x) sin(y) exp(-2νt)
```

Plain English:

The vorticity forms a checkerboard of positive and negative spin. The checkerboard fades over time as viscosity smooths the flow.

## Expected diagnostics

For amplitude 1 and wave number 1:

```text
Energy:    E(t) = π² exp(-4νt)
Enstrophy: Z(t) = 2π² exp(-4νt)
```

Plain English:

Both energy and enstrophy decay exponentially. Enstrophy starts at twice the energy for this setup.

## Numerical reproduction

Parameters:

```text
Grid:       64 x 64
Viscosity:  0.01
dt:         0.001
Steps:      500
Final time: 0.5
```

Observed result:

```text
Numerical energy:      9.67417121
Exact energy:          9.67417314
Energy error:          1.935e-06

Numerical enstrophy:   19.34834241
Exact enstrophy:       19.34834628
Enstrophy error:       3.870e-06

Max vorticity error:   1.980e-07
```

## Timestep convergence

NS Lab also checks that the Taylor-Green numerical error decreases as the timestep decreases.

The convergence test runs the same benchmark to final time `t = 0.5` with:

```text
dt = 0.004
dt = 0.002
dt = 0.001
```

The test verifies that:

```text
error(dt = 0.002) < error(dt = 0.004)
error(dt = 0.001) < error(dt = 0.002)
```

It also checks approximate first-order timestep convergence:

```text
error(dt = 0.004) / error(dt = 0.002) ≈ 2
error(dt = 0.002) / error(dt = 0.001) ≈ 2
```

This is expected because the current solver uses explicit Euler timestepping, which is first-order accurate in time.

Plain English:

Halving the timestep should roughly halve the time-discretization error.

## Interpretation

The numerical solution closely matches the exact Taylor-Green solution at final time.

The remaining error is small and expected because the current solver uses explicit Euler timestepping. This benchmark gives NS Lab its first paper-backed exact-solution reproduction.

## Related code

```text
src/ns_lab/benchmarks/taylor_green.py
examples/02_taylor_green_vortex.py
tests/test_taylor_green_benchmark.py
tests/test_taylor_green_solver_reproduction.py
tests/test_taylor_green_convergence.py
```

## Sources

This benchmark is based on the Taylor-Green / Taylor vortex-array exact solution for incompressible Navier-Stokes flow.

Primary historical source:

- Taylor, G. I. (1923). “On the decay of vortices in a viscous fluid.” *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 46(274), 671–674. DOI: 10.1080/14786442308634295

Modern benchmark/formula reference:

- TensorMesh documentation. “Taylor-Green Vortex (Convergence Study).” Used as a modern reference for the periodic exact-solution benchmark form and verification use case.

## Convention note

Some references write the 2D Taylor-Green velocity field as:

```text
u(x, y, t) = -cos(x) sin(y) exp(-2νt)
v(x, y, t) =  sin(x) cos(y) exp(-2νt)
```

NS Lab currently uses the equivalent rotated/sign-convention form:

```text
u(x, y, t) =  sin(x) cos(y) exp(-2νt)
v(x, y, t) = -cos(x) sin(y) exp(-2νt)
```

The implemented vorticity field is:

```text
ω(x, y, t) = 2 sin(x) sin(y) exp(-2νt)
```

This convention is documented so future reproductions can compare formulas without accidentally treating sign or coordinate-order differences as mathematical disagreements.

## Library record

Machine-readable source card:

```text
references/sources/taylor_1923_decay_of_vortices.yaml
```

This source card records the paper metadata, formulas used, implementation files, validation tests, assumptions, and reproduction status.

## Status

Passed with 48 tests.