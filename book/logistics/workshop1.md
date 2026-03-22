# Day 1 workshop

## ML for solid mechanics and computational homogenization

In this workshop you will:

1. Learn how to use basic tools of Machine Learning (ML) within the context of solid mechanics and materials modelling.
2. Learn to apply MultiLayer Perceptron (MLP) and Convolutional Neural Network (CNN).
3. Learn to regress stress-strain curve in 2D of a hyperelastic material.
4. **Bonus**: learn to regress effective stiffness tensor of arbitrary microstructure from Representative Volume Element (RVE) simulations represented as a BW image.

We start by using machine learning to fit relatively simple hyperelastic material behavior in large-strain regime. This is an illustrative but representative scenario very often encountered when accelerating computational homogenization with ML. Here the ground truth data comes from a simple hyperelastic constitutive model, whereas in practice it would come from micromechanical simulations followed by homogenization. In any case the techniques you see here would be directly applicable.

## Preparations

If you would like to work on Google Colab, download the [GP](https://github.com/EM-ML-2026/hyperelasticity/blob/main/gp.ipynb) and [NN](https://github.com/EM-ML-2026/hyperelasticity/blob/main/nn.ipynb) notebooks, upload them to Colab and add a code block at the top of the notebook with:

```python
!git clone https://github.com/EM-ML-2026/hyperelasticity
%cd hyperelasticity
```

and that is it!

If working locally, first clone the workshop repository:

```bash
git clone https://github.com/EM-ML-2026/hyperelasticity.git
```

Then navigate to the cloned folder and resolve the conda environment:

```bash
conda env create -f environment.yml
conda activate ml-course-day1
```

Make sure your IDE (e.g VS Code) is using the correct environment. Then load the notebooks and get to work!

## Fitting data coming from a Bertoldi-Boyce hyperelastic model

Consider a known explicit Bertoldi-Boyce hyperelastic constitutive law in two-dimensional setting under plane strain assumption, given by its energy density as
$$
W(\boldsymbol{F}) = C_1(I_1-3)+C_2(I_1-3)^2-2C_1\log(J)+\frac{1}{2}K(J-1)^2,
$$
where $\boldsymbol{F}(\vec{X}) = \boldsymbol{I} + (\vec{\nabla}\vec{u}(\vec{X}))^\mathsf{T}$ is the deformation gradient, $\boldsymbol{I} = \sum_{i=1}^3\vec{e}_i\vec{e}_i$ is the identity tensor with respect to a selected Cartesian coordinate basis $\{ \vec{e}_1, \vec{e}_2, \vec{e}_3 \}$, $\vec{\nabla} = \sum_{i=1}^3\frac{\partial}{\partial X_i}\vec{e}_i$ is the gradient operator with respect to the reference configuration, $\vec{u}(\vec{X})$ the displacement field, $I_1 = \mathrm{tr}{\boldsymbol{C}}$ is the first invariant of the right Cauchy-Green deformation tensor $\boldsymbol{C} = \boldsymbol{F}^\mathsf{T}\cdot\boldsymbol{F}$, $\boldsymbol{A}\cdot\boldsymbol{B}$ denotes contraction between two second-order tensors, $J = \det{\boldsymbol{F}}$, and $C_1$, $C_2$, and $K$ are material constants. 

For hyperelastic materials, the corresponding first Piola-Kirchhoff (1PK) stress and its constitutive tangent are obtained from the energy density potential above as
$$
\boldsymbol{P}(\boldsymbol{F}) = \frac{\partial W(\boldsymbol{F})}{\partial\boldsymbol{F}},
$$
$$
^{4}\boldsymbol{D}(\boldsymbol{F}) = \frac{\partial^2 W(\boldsymbol{F})}{\partial\boldsymbol{F}\partial\boldsymbol{F}} = \frac{\partial\boldsymbol{P}(\boldsymbol{F})}{\partial\boldsymbol{F}},
$$
which, in component form, read as
$$
P_{ij} = \frac{\partial W}{\partial F_{ij}} = 2C_1F_{ij}+4C_2(I_1-3)F_{ij}-2C_1F^{-1}_{ji}+K(J-1)JF^{-1}_{ji}
$$
$$
D_{ijkl} = \frac{\partial^2W}{\partial F_{ij}\partial F_{kl}} = \frac{\partial P_{ij}}{\partial F_{kl}} = 2C_1\delta_{ik}\delta_{jl}+8C_2F_{kl}F_{ij}+4C_2(I_1-3)\delta_{ik}\delta_{jl}+
2C_1F^{-1}_{li}F^{-1}_{jk}+KJ^2F^{-1}_{ji}F^{-1}_{lk}+KJ(J-1)F^{-1}_{ji}F^{-1}_{lk}-KJ(J-1)F^{-1}_{li}F^{-1}_{jk}.
$$

The energy density, 1PK stress, and corresponding constitutive stiffness are provided as an explicit function as well as in the available training dataset, expressed as a function of the deformation gradient.

## Exercise 1 — Building Gaussian Process-based surrogate models

Start in `gp.ipynb` and get acquainted with the code. Try to relate it to what you have seen in the lecture. Run the notebook and look at the predictions. As you first open the notebook, a GP model is being trained with just **two** $(\mathbf{F},P_{11})$ pairs, which makes predictions very poor. Then you can work on the following points:

- Gradually increase the number of samples. Does the prediction quality increase? Does the variance in the prediction correlate with the quality of the posterior mean? In other words, does the epistemic uncertainty help you reach good accuracy with as little data as possible? Remember in practice you do not have access to the ground truth, and each data point might be expensive to obtain (e.g from a heavy micromechanical simulation)
- Gradually increase dataset size and look at the hyperparameter values. Do they tend to stabilize as the dataset increases? Play with the `n_restarts` parameter we pass to the `train_hyperparameters()` function. How sensitive the optimized hyperparameters are to the initial guess? Does that change with the size of the dataset ($N$)?
- Look at the predictions for the tangent stiffness. Does the variance in $P_{11}$ correlate with regions with low accuracy in $D$? Does the accuracy in $D$ converge with dataset size at the same rate as the accuracy in terms of $P_{11}$? Remember we are only observing the stresses here. Would there be value in also being able to observe $D$?
- For the initial $N=2$ dataset, predictions of $D_{1112}$ and $D_{1121}$ are spot on even though all other predictions are bad. Can you reason out why?
- Set $N=1000$ to use all samples in the dataset. What happens to the speed of your computations? Compare training speed for the same dataset size with the ones in Exercise 2 and investigate how well GPs scale with dataset size. Looking at the code, which operations do you think form a bottleneck in the computation?

## Exercise 2 — Building neural network-based surrogate models