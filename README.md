# Methods for the Bayesian estimation of free energies
**UNDER DEVELOPEMENT**

Repository to host tools for Bayesian estimate of free energies. 

## Bayesian adaptive mixture sampling (BAMS)
An online method to calculate the ratios of the normalizing constants in mixtures of distributions. The method is inspired by thinking of error estimates for free energies calculated with [Self Adjusted Mixture Sampling](http://www.tandfonline.com/doi/abs/10.1080/10618600.2015.1113975).
The result is a method that is similar in principle to the method by [Bartel and Karplus](https://github.com/choderalab/bayesian-free-energies/blob/master/references/Bartels1997Multidimentional.pdf).

## Installation
This package is still in development, so to install type

```
python setup.py install
```

## Dependencies
* numpy
* scipy
* emcee
* autograd (dependency to be removed soon)

## Directories
`bams/` Contains a tools for Bayesian adjusted mixture sampling

`references/` Papers and references of prior Bayesian free energy work
