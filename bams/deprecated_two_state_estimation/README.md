# Harmonic Oscillator Test
**DEPRECATED**

A preliminary, proof of principle, that expanded ensemble sampling with
Bayesian estimation could be used for an adaptive method. The tools in
this folder only work for systems with exactly two states. A harmonic
oscillator example is explored.

## Contains
* `harmonic_mixture_sampler.py`        Class to sample configurations and states from a mixture of harmonic oscillators
* `free_energy_estimators.py `         Classes estimate free energies via maximum likelihood and Bayesian sampling
* `OnlineBayesianFreeEnergies.ipynb`   Notebook that uses the functions contained in the above to provide an example of the Bayesian method

## Preliminary work
* `Example_FittingFreeEnergies.ipynb`   Example of how to sample states and estimate free energies
* `Example_OpemmHarmonicOscillator.ipynb`   Notebook to easily explore sampling a harmonic oscillator with OpenMM
* `Emcee_test.ipynb` Initial exploration of MCMC with Emcee
