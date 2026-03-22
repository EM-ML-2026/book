# Day 2 workshop

## Benchmark Problems: PINN vs. Neural Green's Operator (NGO)

Both benchmarks concern the steady diffusion equation on the unit square:

$$-\nabla \cdot (\theta\, \nabla u) = f \qquad \text{on } [0,1]^2$$

with appropriate boundary conditions. The goal is to compare two machine-learning approaches for solving this PDE:

- **PINN** — Physics-Informed Neural Network (JAX / Equinox)
- **NGO** — Neural Green's Operator (PyTorch)

---

## Benchmark 1 — Homogeneous coefficients, analytical solution

The diffusion coefficient is constant ($\theta = 1$), reducing the PDE to the classical Poisson equation. The exact solution is

$$u(x,y) = e^{xy},$$

which gives the forcing term $f(x,y) = -e^{xy}(x^2 + y^2)$.

Both models are evaluated on this same problem. Since the exact solution is known analytically, the relative $L^2$ error can be computed directly.

---

## Benchmark 2 — Heterogeneous coefficients, GRF manufactured solution

Both $\theta$ and $u$ are realisations drawn from a Gaussian Random Field (GRF). The forcing $f$ and all boundary data are derived analytically from $\theta$ and $u$ via the manufactured-solution principle provided by the NGO repository, so the exact solution is known by construction.

The NGO is trained on many such random samples and then tested on a single instance. The PINN is applied to that same instance, with $\theta$ and $f$ provided as grid data from the NGO example. This benchmark tests how well each method generalises to spatially varying, randomly generated coefficients.

---

## What to compare

For each benchmark, report:

1. The relative $L^2$ error: $\|\hat{u} - u_{\text{exact}}\| / \|u_{\text{exact}}\|$
2. A plot of the predicted solution, the exact solution, and the pointwise error.
3. *(For the PINN)* the training loss curve.

Consider the following questions:

- Which method is more accurate on the analytical benchmark?
- How does accuracy change when moving to heterogeneous coefficients?
- What are the computational trade-offs (training time, inference time)?
