import numpy as np
from scipy import optimize
from scipy import special

class MaximumLikelihood(object):
    """
    Class to calculate the ratio of normalising constant from bayesian mixture sampling of a two state system.

    Can either calculate the free energies via least squares fitting or maximum likelihood.
    """

    def __init__(self, zetas, nsuccesses, nsamples):
        """
        Initialise the fitting tool.

        Parameter
        ---------
        zetas : numpy array
          the biasing potential applied at each mixture sampling run
        nsuccesses : int or numpy array
          the number of times the second state was sampled for each zeta
        nsamples : int or numpy array
          the number of samples drawn at each simulation. If integer, assumed to be the same for all simulations
        """
        self.zetas = zetas
        self.nsuccesses = nsuccesses
        if type(nsamples) == int:
            self.nsamples = np.repeat(nsamples, len(zetas))
        else:
            self.nsamples = nsamples

        self.proportions = self.nsuccesses / self.nsamples

    def logistic(self, f):
        """
        Logistic function

        Parameters
        ----------
        f : float
          the logarithm of the ratio of normalising constants

        Return
        ------
        numpy array
          the output of the logistic function

        """
        return 1/ (1 + np.exp(f - self.zetas))

    def _chooseln(self, N, k):
        """
        The log of binomial coefficient. Useful for evaluating log likelihoods for binomial models.

        Parameters
        ----------
        N : float
          the number of trials
        k : float
          the number of successes

        Return
        ------
        float
          the log of the binomial coefficient
        """
        return special.gammaln(N+1) - special.gammaln(N-k+1) - special.gammaln(k+1)

    def log_likelihood(self, f_guess):

        p_guess = self.logistic(f_guess)
        q_guess = 1.0 - p_guess

        l = np.sum([self._chooseln(N, k) for (N, k) in zip( self.nsamples, self.nsuccesses ) ])
        l += np.sum(self.nsuccesses * np.log(p_guess) + (self.nsamples - self.nsuccesses) * np.log(q_guess))

        return l

    def sum_of_squares(self, f_guess):
        """
        Return the sum of squares of the normalised logistic function and a guess of the free energy

        Parameters
        ----------
        f_guess : float
          a guess of the free energy difference between states

        Returns
        -------
        the sum of squares of the predicted state proportions
        """
        predicted = self.logistic(f_guess)
        return np.sum( (self.proportions - predicted)**2 )

    def least_squares(self, f_guess = None):
        """
        Minimise the sum of squares of the state proportions to calculate free energy

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
            f_guess = np.random.choice(self.zetas, size = 1)

        fit = optimize.minimize(self.sum_of_squares, f_guess, method='BFGS')

        return fit.x

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
            f_guess = np.random.choice(self.zetas, size=1)

        fit = optimize.minimize(lambda x: -self.log_likelihood(x), f_guess, method='Powell')

        return fit.x