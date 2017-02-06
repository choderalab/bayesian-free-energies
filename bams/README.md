# Bayesian Adjusted Mixture Sampling
Tools and examples for adaptive Bayesian estimation of free energies in
expanded ensembles. For a detailed description of what this package is 
designed for, please see `theoretical_notes/BAMS in brief.ipynb`.

The classes `MultinomialBayes` and `BayesAdaptor` contained in `bayes_adaptive_fitter.py` provides Bayesian
methods for the estimation of free energies using histogram data from simulations

## Quick examples 
### bayes_adaptive_fitter.MultinomialBayes
The class `MultinomialBayes` calculates the relative free energies of 
states using the number of times a system visited those states. Say,
for example, that the state state space of system is divided into 4. 
During an unbiased simulation the system visits each of the for states the
following number of times:
```
counts = np.array((10, 103, 243, 82))
```
It is assumed that these counts are from independent samples. We'll note
the fact that the system was unbiased by recording the exponentiated 
weights applied to each state:

```
weights = np.array((0.0, 0.0, 0.0, 0.0))
```
With this information, we can estimate the relative free energies using
Bayesian inference. The free energies will be relative the first state.
We'll choose the prior for the free energies to be broad Gaussians:
```
fitter = MultinomialBayes(zetas=zetas, counts=counts, prior='gaussian', location=0, spread=50)
```
We can now estimate the maximum a posterior (MAP) estimate for the free
energies:
```
print fitter.map_estimator()
```
We can go further and sample from the posterior with the package `emcee`:
```
samples = fitter.sample_posterior()
```
## Dependencies
* numpy
* scipy
* emcee
* autograd (dependency to be removed soon)

## Contents
### Files
```
bayes_adaptive_fitter.py    Classes for adaptive Bayesian inference of free energies
```

```
adaptive_tools.py           Miscellaneous tools for examples and plotting.
```
### Folders
```
gaussian_mixture_example/    Demonstration of bayes_adaptive_fitter to Gaussian mixtures
```

```
depracted_two_state_estimation/  Initial proof of principle with a two state harmonic oscillator
```

