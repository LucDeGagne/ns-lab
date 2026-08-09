# Benchmark 001: Taylor-Green Vortex

## Summary

This benchmark reproduces the 2D Taylor-Green vortex, an exact incompressible Navier-Stokes solution on a periodic domain.

The goal is to verify that NS Lab can:

- represent the exact velocity and vorticity fields
- evolve the vorticity numerically
- compare the numerical result against the exact solution
- track energy and enstrophy decay

## Domain

The benchmark uses the default periodic domain:

```text
[0, 2π] x [0, 2π]
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

## Library record

Machine-readable source card:

```text
references/sources/taylor_1923_decay_of_vortices.yaml