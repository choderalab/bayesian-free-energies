import numpy as np
from scipy import optimize
from scipy import special
import emcee

class MaximumLikelihood(object):
    """
    Class to calculate the ratio of normalising constant from bayesian mixture sampling of a two state system. The
    sampling distribution of the data is bimonial. Can either calculate the free energies via least squares fitting or
    maximum likelihood, the former may be more stable in some cases.

    This class works in conjunction with the HarmonicSwapper class.

    Example
    -------
    Perform mixture sampling then estimate the free energy via maximum likelihood.
    Samples states:
    >>> zetas = range(-5,5)
    >>> n_success = []
    >>> for z in zetas:
    >>>    Swapper = HarmonicSwapper(zeta = [0.0, z])
    >>>    Swapper.mixture_sample(openmm = False)
    >>>    n_success.append( swapper.state_counter )
    Estimate free energy difference:
    >>> Fitter = MaximumLikelihood(zetas = np.array(zetas), nsuccesses = np.array(n_success), nsamples = swapper.nmoves)
    >>> Fitter.max_like()
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
        """
        The logarithm of the likelihood for the binomial data. The normalising constant is not necessary for the
        purpose of maximum likelihood estimation, or MCMC sampling from the posterior.

        Parameter
        ---------
        f_guess : float
          the free energy estimate
        """

        p_guess = self.logistic(f_guess) - 1E-10
        q_guess = 1 - p_guess     # Ensuring that I don't get zeros for log(q_guess)

        l = np.sum(self.nsuccesses * np.log(p_guess) + (self.nsamples - self.nsuccesses) * np.log(q_guess))

        # To include the normalising constant:
        #l += np.sum([self._chooseln(N, k) for (N, k) in zip( self.nsamples, self.nsuccesses ) ])

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


class BayesianSampler(MaximumLikelihood):
    """
    Class to estimate the free energy difference between two states using the 'emcee' package to step from
    the posterior. Currently, only prior distributions with 2 parameters are supported.

    Example
    -------
    Perform mixture sampling then step from the posterior of the free energy difference.
    Samples states:
    >>> zetas = range(-5,5)
    >>> n_success = []
    >>> for z in zetas:
    >>>    Swapper = HarmonicSwapper(zeta = [0.0, z])
    >>>    Swapper.mixture_sample(openmm = False)
    >>>    n_success.append( Swapper.state_counter )
    Estimate free energy difference:
    >>> Sampler = BayesianSampler(zetas = np.array(zetas), nsuccesses = np.array(n_success), nsamples = Swapper.nmoves)
    >>> chain = Sampler.sample_posterior()
    >>> samples = chain[:, 50:, :].reshape((-1, 1))
    >>> print( flat_samples.mean() )
    >>> print( flat_samples.std() )
    """
    def __init__(self, zetas, nsuccesses, nsamples, prior = 'normal', location = 0, spread = 5):
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
        prior : str
          the name of the prior distribution that will be used.
        location : float
          the location parameter of the prior, e.g. for 'normal', 'location' is the mean
        spread : floar
          the spread parameter of the prior, e.g. for 'normal', 'spread' is the standard deviation
        """
        self.zetas = zetas
        self.nsuccesses = nsuccesses
        if type(nsamples) == int:
            self.nsamples = np.repeat(nsamples, len(zetas))
        else:
            self.nsamples = nsamples
        self.proportions = self.nsuccesses / self.nsamples

        # parameters for the prior distribution.
        self.prior = prior
        self.location = location
        self.spread = spread

    def log_prior_normal(self,f):
            """
            The log of prior for the normal distribution. The normalising constant is not required.

            Parameter
            ---------
            f : float
              the log of the ratio of normalising constants

            """
            return -( (f - self.location) ** 2) / (2.0 * self.spread ** 2)

    def sample_posterior(self, nwalkers = 50, nmoves = 500):
        """
        Sample from the posterior using the Emcee package. The initial starting point for the sampler is the
        least squares fit.

        nwalkers : int
          the number of walkers for the 'affine invarient sampler'
        nmoves : int
          the number of moves to perform with the 'affine invarient sampler'

        Returns
        -------
        chain : numpy.ndarray
          the output from the Emcee sampler
        """
        if self.prior == 'normal':
            def log_posterior(f):
                return self.log_likelihood(f) + self.log_prior_normal(f)

        #TODO: change intial guess to MAP estimate.
        # Initialise the walkers
        initial_f = self.least_squares()[0]
        initial_positions = [initial_f + 1e-1*np.random.randn(1) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, 1, log_posterior)
        sampler.run_mcmc(initial_positions , nmoves)

        return sampler.chain