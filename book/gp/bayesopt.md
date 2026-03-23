# Bayesian Optimization

We can also use GPs to derive one of the most powerful optimization algorithms being used today. Imagine you are trying to **find the maximum of a black-box function**, and you can only evaluate this function a handful of times (e.g because each evaluation is an expensive experiment or a model evaluation that takes long to run). We have to pick our points in a very goal-oriented way, and the **epistemic uncertainty** given by GPs can be a huge help. Try it out yourself in this interactive plot, where you are only allowed to sample 5 times:

<iframe src="../_static/gp-bayes-opt.html?mode=basic" width="100%" height="680" frameborder="0"></iframe>

Did you start by (i) putting one or two observations at random and then (ii) focusing on regions with high mean values while not forgetting regions of high uncertainty? Then you get the simple intuition behind Bayesian Optimization. The idea is to find the maximum (or minimum) of a function by balancing two mechanisms:

- **Exploration**: We want to put observations at regions in input space we do not know much about. This is represented by a high GP variance
- **Exploitation**: We want to pinpoint our peak as close as possible, so we also want to observe the region around our current optimum $f^*$

Instead of doing the above by eye (which is anyway impossible for more than 2 input features), Bayesian Optimization does it by computing a so-called **Acquisition Function** over the whole domain and letting it suggest the best next point to sample. A popular acquisition function is the **Expected Improvement (EI)**:

$$
EI(x) = \mathbb{E}[\mathrm{max}(f(x)-f^*,0)]
$$

Since our function $f(x)$ is given by a GP, computing this expectation will naturally balance regions with a high mean but low variance and low mean but high variance. Computing the EI everywhere is cheap because we only need to evaluate the GP everywhere, not the actual black-box objective function. We then just pick the point $x$ that corresponds to the highest EI, compute the real function there and update our GP estimate. You can play with the interactive plot again, but now we give you the EI and its peak (dashed red line):

<iframe src="../_static/gp-bayes-opt.html?mode=ei" width="100%" height="900" frameborder="0"></iframe>

Was it easier to find the peak this time?

## Applications of Bayesian Optimization

BO can in principle be used for any optimization problem. Although we only wanted to give you a general idea, the method is an active field of research and many improvements to the basic formulation above have already been proposed. One of the most popular applications of BO is to perform **hyperparameter tuning** in frequentist models such as neural nets. This is at the core of the popular [Optuna](https://optuna.org/) package, where the objective function to be minimized is a validation loss and the design variables are the hyperparameters in the model (e.g network architecture, regularization strength, etc).


