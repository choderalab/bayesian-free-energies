import numpy as np
from scipy import optimize
from scipy import special
import emcee

class MultinomialBayes(object):
    """
    Class to estimate free energies from multinomial samples via Bayesian estimation
    """
    def __init__(self, zetas, counts):
        """
        Parameters
        ----------
        zetas: numpy.ndarray
          the biasing potential applied to each state
        counts: numpy.ndarray
          the number of times the state was visited
        """

        self.zetas = zetas - zetas[0]     # Setting the first zeta to zero
        self.counts = counts

    def logistic(self,f,z):
        """
        The logistic function:

            g(x) = 1 / (1 + exp(f - z))

        Parameters
        ----------
        f: float
          the point of inflection (free energy)
        z: numpy.ndarray or float
          the independent variable (biasing potential)

        Returns
        -------
        the value of the logistic function for the specified parameters
        """
        return 1 / (1 + np.exp(f - z))

    def loglikelihood(self, f):
        """
        The log likelihood of the counts, with the normalisation constant discarded:

        Parameter
        ---------
        f: numpy.ndarray
          the vector of estimates for the free energy

        Returns
        -------
        l: float
          the log of the unnormalised likelihood
        """
        diffs = self.zetas - f
        l = self.counts * diffs + self.counts * np.log(np.sum(np.exp(diffs)))
        l = np.sum(l)

        return l

    def max_like(self, f_guess=None):
        """
        Find the free energy that maximises the likelihood of observing the state labels

        Parameter
        ---------
        f_guess : float
          initial guess of the free energy difference

        Returns
        -------

        f_optimised : float
          the fitted free energy difference
        """
        if f_guess is None:
            f_guess = self.zetas + np.random.normal(size = len(zetas))
            f_guess[0] = 0.0


        fit = optimize.minimize(lambda x: -self.log_likelihood(x), f_guess, method='Powell')

        return fit.x