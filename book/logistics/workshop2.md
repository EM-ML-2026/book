# Day 2 workshop

## ML for solving partial differential equations

In this workshop you will:

1. Learn how to use physics-informed neural networks and neural operators to solve simple PDEs.
2. Learn to change boundary conditions and PDE coefficients.
3. Learn to benchmark solutions from PINNs and Neural Green’s Operators.

You will work in pairs to test and compare the performance of two different machine learning approaches for the solving Poisson equation: Physics-Informed Neural Networks (PINNs) and Neural Green's Operators (NGOs). The goal is to understand the strengths and weaknesses of each method in terms of accuracy, generalization, and computational efficiency.

## Preparations

If working locally, clone the [simple-pinns](https://github.com/EM-ML-2026/simple-pinns) and [ngo](https://github.com/EM-ML-2026/ngo) repositories. Then navigate to the cloned folder and resolve the virtual environment as follows:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you would like to work on Google Colab, add a code block at the top of the notebook with:

PINN:

```python
!git clone https://github.com/EM-ML-2026/simple-pinns.git
%cd simple-pinns
!pip install -r requirements.txt
!python 02_2d_steady_poisson.py
```

NGO:

```python
!git clone https://github.com/EM-ML-2026/ngo.git
%cd ngo
!pip install -r requirements.txt
!pip install -e .
%run examples/steadydiffusion.ipynb
```

Make sure your IDE (e.g VS Code) is using the correct environment. Then load the scripts and get to work!

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
