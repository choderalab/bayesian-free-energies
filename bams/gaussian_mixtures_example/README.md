# Expanded ensemble of Gaussian Mixtures

Demonstration of adaptive Bayesian estimation of free energies using 
mixture distributions of 1-dimensional Gaussians.

## Contains

```
gaussian_mixture_sampler.py         Contains class for Gaussian mixture sampling
```

The distributions that can be sampled with this tool are the following form

$$ p(x, l=i| \zeta) = \frac{q_i(x)\exp(\zeta_i)}{\sum^K_{j=1}\exp(\zeta_j - f_j)} $$

where $i \in {1,2,...,K)$, $q_i(x)$ is an unnormalized 1-dimensional Gaussian, $K$ is the number
of Gaussians in the mixture, $\zeta_i$ is the exponential weight/biasing
potential of the $i$th state and $f_i$ is the free energy of the $ith$ 
state relative to $f_1$.

```
Example_Multistate_Inference.ipynb      Example of mixture sampling and Baysian inference 
```

```
Example_Multistate_Inference.ipynb      Example of adaptive Baysian mixture sampling
```