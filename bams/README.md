# Bayesian Adjusted Mixture Sampling
Tools and examples for adaptive Bayesian estimation of free energies in
expanded ensembles.

## Contains
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


## Notes
The tools and examples contained here, particularly the 
`BayesianAdaptor` class, are designed to adaptively infer the relative ratios 
of normalizing constants within expanded ensembles (i.e. mixture distributions)
of comprised of $K$ distributions that are in the following form:

$$ p(x, l=i| \zeta) = \frac{q_i(x)\exp(\zeta_i)}{\sum^K_{j=1}Z_j\exp(\zeta_j)} $$

where $x$ is the configuration, $i$ the state, $\zeta_i$ is a user
defined biasing potential of the $i$th state, $q_i(x)$ is the 
unnormalized density of the $i$th distribution, and $Z_i$ is the 
respective normalizing constant. We are concerned with estimating the 
ratios $Z_i/Z_0 \, i \neq 0$. 

By defining the free energy of the $i$th state as

$$ f_i = -\ln(Z_i) $$

we can write the marginal of $p(x, l=i| \zeta)$ over $x$ as 

$$ p(l=i | \zeta) = \frac{\exp(\zeta_i - f_i)}{\sum_{j=1}^{K} \exp(\zeta_j - f_j)} $$

This marginal is a multinomial distribution over the states, which 
allows for Bayesian inference of the $f_i$s (and thus $Z_i$) relative to
the 0th state. The `MultinomialBayes` class is designed for inference of
free energies with such distributions, and takes as input the counts in 
each state and the applied biases to return Bayesian estimates of the relative $f_i$s.
