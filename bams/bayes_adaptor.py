import numpy as np
from scipy import optimize
from scipy.misc import logsumexp
import emcee
from copy import deepcopy

class MultinomialBayesEstimator(object):
    """
    Class to estimate free energies from multinomial mixture samples via Bayesian estisation.

    The tools in this class can calculate the ratios of normalizing constants for mixtures of the following
    form:

        p_i(x) = q_i(x) * exp(zeta_i) / sum_i[Z_i * exp(zeta_i)],

    where x is the configuration, q_i(x) is the unnormalized density of the ith distribution, Z_i is normalizing
    constant of the ith distribution and zeta_i is the exponentiated weight of the ith element. We are interested
    in calculating the ratios Z_i/Z_j.

    With a method that can step over the states and configurations of p_i(x), this class can estimate the
    logarithm of the ratios Z_i/Z_j using the counts for the number of times each state was visited by the
    sampler.

    This is because the marginal over the configurations, given by

        p_i = Z_i * exp(zeta_i) / sum_i[Z_i * exp(zeta_i)],

    is a multinomial distribution over the states, such that the likelihood function for the normalizing constants
    is known exactly. This is the idea behind this tool.

    Examples
    --------
    A mixture sampler provides the following counts for each state
    >>> counts = np.array((10, 103, 243, 82))

    It is assumed that these counts are from independent samples. These samples were generated with
    the following exponentiated biases applied to each state
    >>> zetas = np.array((10.0, 30.0, 43.0, 28.0))

    With this information, we can estimate the logarithm of each normalizing constant (free energy) up to an factor.
    Being a Bayesian method, we must provide prior information on what the free energies. Let's choose broad Gaussians:
    >>> fitter = MultinomialBayesEstimator(zetas=zetas, counts=counts, prior='gaussian', location=0, spread=100)

    Estimate the MAP estimate for the free energies
    >>> print fitter.map_estimator()

    Sample from the posterior with emcee:
    >>> samples = fitter.sample_posterior()

    Analyze to your hearts content.

    Multiple samples with different biases can also be added by stacking the data, i.e
    >>> counts = np.vstack(((10, 103, 243, 82), (30, 11, 23, 72)))
    >>> zetas = np.vstack(((10.0, 30.0, 43.0, 28.0), (4.0, 14.0, 36.0, 81.0)))
    """
    def __init__(self, counts, zetas=None, prior='gaussian', location=None, spread=None):
        """
        Parameters
        ----------
        counts: numpy.ndarray or list
            the number of times the state was visited for a number of repeats, where the columns correspond to the states
            and rows repeats.
        zetas: numpy.ndarray or list
            the biasing potentials applied to each state, where the columns correspond to the states and rows repeats.
        prior: string
            the name of the prior distribution. Choice is between 'gaussian', 'laplace', and 'cauchy'
        location: float, numpy.ndarray
            the location parameter of the prior for each free energy, e.g. the mean for 'gaussian'. If float, the same
            value is applied to all free energies.
        spread: float, numpy.ndarray
            the spread parameter of the prior for each free energy, e.g. the standard deviation for 'gaussian'.
            If float, the same value is applied to all free energies.
        """

        # Carefully formatting the histogram count matrix so that rows match the applied biasing potentials
        if type(counts) == list:
            self.counts = np.array(counts)
        elif type(counts) == np.ndarray:
            if counts.ndim == 1:
                self.counts = np.array([counts])
            else:
                self.counts = counts
        else:
            raise Exception('The histogram counts must have the format of either a list or a numpy array')

        # Formatting the bias potentials to match the counts. The (i,j) element of the count matrix should be the
        # counts that resulted from the application of the (i,j)th bias.
        if zetas is None:
            self.zetas = np.zeros(self.counts.shape)
        elif type(zetas) == list:
            self.zetas = np.array(zetas)
        elif type(zetas) == np.ndarray:
            if zetas.ndim == 1:
                self.zetas = np.array([zetas])
            else:
                self.zetas = zetas
        else:
            raise Exception('The biases must have the format of either a list or a numpy array')

        if self.zetas.shape != self.counts.shape:
            raise Exception('Error: the dimensions of the biasing potentials and state counts must match')

        # Pre-assignment to save the posterior samples
        self.samples = []

        # Set the initial guess of the free energies to the relative counts
        self.free_energies = np.sum(self.zetas - np.log(self.counts + 0.1), axis=0)
        self.free_energies -= self.free_energies[0]

        # parameters for the prior distribution
        if prior.lower() in ('gaussian', 'laplace', 'cauchy'):
            self.prior = prior
        else:
            raise Exception('The prior must be either "gaussian", "laplace", or "cauchy".')
        if location is None:
            self.location = np.zeros(len(self.free_energies))
        elif type(location) == float:
            self.location = np.repeat(location, len(self.free_energies))
        else:
            self.location = location
        if spread is None:
            # Assuming no correlation structure between free energies
            self.spread = np.repeat(5, len(self.free_energies))
        elif type(spread) == float:
            self.spread = np.repeat(spread, len(self.free_energies))
        else:
            self.spread = spread

    def _expected_counts(self, f):
        """
        Predict the counts for each state and for each biasing potential. Used to calculate the sum of squares.

        Parameters
        ----------
        f: numpy.ndarray
            Estimate of the free energies for each state relative to the first (f[0])

        Returns
        -------
        expected_counts: numpy.ndarray
            The predicted counts for each state, in the same dimensions as self.counts
        """
        p = np.exp(self.zetas - f)
        row_sums = np.sum(p, axis=1)
        p_norm = p / row_sums[:, np.newaxis]
        row_sums = np.sum(self.counts, axis=1)
        expected_counts = p_norm * row_sums[:, np.newaxis]

        return expected_counts

    def _log_prior_gaussian(self, f):
        """
        The log of prior for the normal distribution. The normalising constant is not required.

        Parameter
        ---------
        f : float
            the log of the ratio of normalising constants

        """
        return -np.sum(((f - self.location) ** 2) / (2.0 * self.spread ** 2))

    def _log_prior_laplace(self,f):
        """
        The log of prior for the laplace distribution. The normalising constant is not required.

        Parameter
        ---------
        f : float
            the log of the ratio of normalising constants

        """
        return -np.sum(np.absolute(f - self.location) / self.spread)

    def _log_prior_cauchy(self,f):
        """
        The log of prior for the cauchy distribution. The normalising constant is not required.

        Parameter
        ---------
        f : float
            the log of the ratio of normalising constants

        """
        #return -np.sum(np.log((f - self.location)**2 - self.spread**2))
        return -np.sum(np.log(1 + ((f - self.location)/self.spread)**2))

    def _log_likelihood(self, f, scale=1.0):
        """
        The log likelihood of the counts, without terms proportional to the free energy.
        The normalisation constant discarded.

        Parameter
        ---------
        f: numpy.ndarray
            the vector of estimates for the free energy
        scale: float
            the factor with which to diminish the counts due to correlated samples

        Returns
        -------
        l: float
          the log of the unnormalized likelihood
        """
        rn = np.sum(self.counts, axis=0) * scale    # Sum of the counts across the repeats at each zeta index
        zn = np.sum(self.counts, axis=1) * scale    # Sum of the counts across the zetas at each repeat
        l = -np.sum(rn * f) - np.sum(zn * logsumexp(self.zetas - f, axis=1))

        return l

    def map_estimator(self, f_guess=None, method='BFGS'):
        """
        Provides a maximum a posteriori estimate of the free energies relative to state 0.

        Parameters
        ----------
        f_guess: numpy.ndarray
            Vector of initial values of the free energies of each state. The first element will be ignored as free
            energies will be relative to the first state
        method: str
            the 'method' parameter used in optimize.minimize

        Returns
        -------
        fit.x: numpy.ndarray
            vector of free energies relative to the first state
        """
        # Defining an internal log_posterior function to minimize for fitting.
        if self.prior == 'gaussian':
            def loss(f):
                """
                The negative log of the posterior with Gaussian priors on the free energies
                """
                return -self._log_likelihood(np.hstack((0.0, f))) - self._log_prior_gaussian(np.hstack((0.0, f)))
        elif self.prior == 'laplace':
            def loss(f):
                """
                The negative log of the posterior with Laplace priors on the free energies
                """
                return -self._log_likelihood(np.hstack((0.0, f))) - self._log_prior_laplace(np.hstack((0.0, f)))
        elif self.prior == 'cauchy':
            def loss(f):
                """
                The negative log of the posterior with Cauchy priors on the free energies
                """
                return -self._log_likelihood(np.hstack((0.0, f))) - self._log_prior_cauchy(np.hstack((0.0, f)))
        else:
            raise Exception('The prior "{0}" is not supported'.format(self.prior))

        if f_guess is None:
            f_guess = deepcopy(self.free_energies)
            f_guess -= f_guess[0]

        fit = optimize.minimize(loss, f_guess[1:], method=method)

        self.free_energies = np.hstack((0.0, fit.x))

        return fit.x

    def sample_posterior(self, nwalkers=50, nmoves=500, f_guess=None):
        """
        Sample from the posterior using the Emcee package. The initial starting point for the sampler is the either the
        current estimate of the free energies or MAP estimate.

        nwalkers: int
            the number of walkers for the 'affine invariant sampler'
        nmoves: int
            the number of moves to perform with the 'affine invarient sampler'
        f_guess: numpy.ndarray
            initial guess of the free energies of each state

        Returns
        -------
        chain : numpy.ndarray
          the output from the Emcee sampler. The dimensionality of the sampled free energies is one less than the number
          of states because the free energy of the first state is set to zero.
        """

        # Defining an internal log_posterior function to minimize for fitting.
        if self.prior == 'gaussian':
            def log_posterior(f):
                """
                The negative log of the posterior with Gaussian priors on the free energies
                """
                return self._log_likelihood(np.hstack((0.0, f))) + self._log_prior_gaussian(np.hstack((0.0, f)))
        elif self.prior == 'laplace':
            def log_posterior(f):
                """
                The negative log of the posterior with Laplace priors on the free energies
                """
                return self._log_likelihood(np.hstack((0.0, f))) + self._log_prior_laplace(np.hstack((0.0, f)))
        elif self.prior == 'cauchy':
            def log_posterior(f):
                """
                The negative log of the posterior with Cauchy priors on the free energies
                """
                return self._log_likelihood(np.hstack((0.0, f))) + self._log_prior_cauchy(np.hstack((0.0, f)))
        else:
            raise Exception('The prior "{0}" is not supported'.format(self.prior))

        # Initialise the walkers
        if f_guess is None:
            self.map_estimator()
            f_guess = deepcopy(self.free_energies)

        # Distribute the walkers in a Gaussian ball within roughly 20% of the initial guess
        scale = np.absolute(f_guess[1:])*0.2
        initial_positions = [f_guess[1:] + scale * np.random.randn(len(f_guess) - 1) for i in range(nwalkers)]
        # Note that the number of free parameters is len(f_guess - 1) as free energies are relative to first.

        sampler = emcee.EnsembleSampler(nwalkers, len(f_guess) - 1, log_posterior)
        sampler.run_mcmc(initial_positions, nmoves)

        self.samples = sampler.chain

        return sampler.chain


class BayesAdaptor(MultinomialBayesEstimator):
    """
    Class to generate biases and free energies estimates from multinomial samples. To be used when
    iterating between estimating free energies via expanded ensemble sampling and running new simulations
    with new biases.

    Example
    -------
    Let's imagine we've run an unbiased simulation over states and configurations. We thin the trejector to produce
    approximately uncorrelated samples in state space, and observe the counts in the histogram over the states
    >>> counts  = np.array((10, 103, 200))

    Clearly there are 3 states in the system. We seek to simultaneously estimate the free energy for each state and
    produce biases that can be applied to an upcoming simulation in order to maximally reduce the uncertainty on the
    current estimates of the free energies. This class handles that task.

    First, we must initialize the class with the above histogram counts, our prior information of the relative free
    energies and the type of Bayesian update scheme we want.
    >>> adaptor = BayesAdaptor(counts=counts, method='map', prior='gaussian', spread=100.0, location=0.0)

    In the above, we've specified that we want our adaptor to choose biases based on MAP estimate of the free energy.
    We've also placed a Gaussian prior of mean 0 and standard deviation of 100 for our relative free
    energies: this is equivalent to saying that we're 95% sure that the true free energies are between -200 and +200
    and tightly peaked around zero. If our previous work indicated that the prior should be 'flatter', we can choose
    prior='laplace' or even prior='cauchy'.

    Let's sample from the posterior, get the MAP estimate, and produce biases for the next set of simulations:
    >>> new_bias = adaptor.update(sample_posterior=True)

    We've used to the flag sample_posterior=True to indicate that we wish to sample from the posterior distribution.
    The MAP estimate is obtained via gradient decent, but by specifying that we want sample from the posterior ensures
    that the starting point for gradient decent is the posterior mean.

    As we've set method='map', new_bias is simulatanoesly an estimate for the free energy *and* that bias that should
    be applied in the next simulation.

    If we wanted to do thompson sampling to pick our next bias, i.e.
    >>> adaptor.method = 'thompson'
    >>> new_biases = adaptor.update()

    Then new_biases are not good estimates of the free energies, either the MAP or mean will be better:
    >>> mean_estimates = np.hstack((0.0, np.mean(adaptor.flat_samples, axis=0)))

    The estimated free energies are all calculated to the first state, which is zero and not sampled over. This is why
    in the above we've added a 0.0 to the vector of means. The MAP estimate based on the mean estimates is then:
    >>> map_estimates = adaptor.map_estimator(f_guess=mean_estimates)

    The point of using method='thompson' is that it may result in estimates of lower variance in fewer iterations of
    sampling and bias generation than method='mean' or method='map'.

    Now let's say we applied our new_bias to our simulation. We thin the data and observe new counts
    >>> new_counts = np.array((49, 25, 33))

    We should update our adaptor with this new information, along with the bias we applied:
    >>> adaptor.zetas = np.vstack((adaptor.zetas, new_biases))
    >>> adaptor.counts = np.vstack((adaptor.counts, new_counts))

    The next bias for the next simulation can then be estimated:
    >>> new_bias = adaptor.update()

    Apply these biases to the next simulation, count the number of times each state was visited and repeat the adaptive
    procedure.
    """
    def __init__(self, counts, zetas=None, method='thompson', logistic=False, prior='gaussian', location=None,
                 spread=None):
        """
        Parameters
        ----------
        counts: numpy.ndarray
            the number of times the state was visited for a number of repeats, where the columns correspond to the states
            and rows repeats.
        zetas: numpy.ndarray
            the previous biasing potentials applied to each state, where the columns correspond to the states
            and rows to the repeats.
        method: string
            the method used to generate new biases for adaptive free energy estimation.
            Either 'thompson', 'map', 'mean', or 'median'.
        logistic: bool
            whether to convolute biasing potential generation with a logistic distribution
        prior: string
            the name of the prior distribution. Choice is between 'gaussian', 'laplace', and 'cauchy'
        location: float, numpy.ndarray
            the location parameter of the prior for each free energy, e.g. the mean for 'gaussian'. If float, the same
            value is applied to all free energies.
        spread: float, numpy.ndarray
            the spread parameter of the prior for each free energy, e.g. the standard deviation for 'gaussian'.
            If float, the same value is applied to all free energies.
        """

        super(BayesAdaptor, self).__init__(counts=counts, zetas=zetas, prior=prior, location=location, spread=spread)

        # The processed posterior samples
        self.flat_samples = np.array(0)

        # The method used to draw the next estimate/bias for the free energies
        method_set = ('thompson', 'map', 'mean', 'median')
        if method.lower() not in method_set:
            raise Exception('Must select a method must be from {0}'.format(method_set))
        else:
            self.method = method

        # Whether convolute estimates/bias generation method with a logistic distribution
        self.logistic = logistic

    def _flatten_samples(self, burnin=100):
        """
        Take samples from the Emcee generated posterior samples and collapse the walkers into a single vector of samples

        Parameters
        ----------
        samples: numpy.ndarray
            samples generated by Emcee via self.sample_posterior()
        burnin: int
            the number of initial samples from each walker to discard

        Returns
        -------
        flat_samples: numpy.ndarray
            Posterior samples for each free energy
        """
        #TODO: add automatic equilibrium detection

        self.flat_samples = self.samples[:, burnin:, :].reshape((-1, len(self.free_energies) - 1))
        return self.flat_samples

    def _gen_biases(self, samples=None):
        """
        Generate biasing potentials given a set of free energy samples for adaptive sampling.

        Note
        ----
        The methods 'mean', and 'median' are the respective averages of the marginal distribution of the
        free energies. On the other hand, the 'thompson' sampling method draws a vector from the full posterior, and
        'map' is the maximum a posteriori estimate from the full posterior.

        Parameters
        ----------
        method: str
            the manner in which the biases are generated
        burnin: int
            the number of initial samples to discard as burn-in
        logistic: bool
            whether to pass the  logistic sampling  biases as the location
        Returns
        -------
        biases: numpy array
            The biasing potentials to apply to each state to achieve the target probabilities
        """

        if self.method == 'thompson':
            if samples is None:
                raise Exception('the method "{0}" must require samples from posterior.'.format(self.method))
            index = np.random.choice(samples.shape[0])
            biases = samples[index, :]
        elif self.method == 'median':
            if samples is None:
                raise Exception('the method "{0}" must require samples from posterior.'.format(self.method))
            biases = np.percentile(samples, 50, axis=0)
        elif self.method == 'mean':
            if samples is None:
                raise Exception('the method "{0}" must require samples from posterior.'.format(self.method))
            biases = np.mean(samples, axis=0)
        elif self.method == 'map':
            if samples is not None:
                f_guess = np.hstack((0.0, np.percentile(samples, 50, axis=0)))
                biases = self.map_estimator(f_guess=f_guess)
            else:
                biases = self.map_estimator()
        else:
            raise Exception('Update method {0} not recognized'.format(self.method))

        if self.logistic:
            biases = np.array([np.random.logistic(loc=f, scale=1, size=1)[0] for f in biases])

        return np.hstack((0.0, biases))

    def update(self, sample_posterior=True, nwalkers=20, nmoves=200, burnin=None):

        if burnin is None:
            burnin = int(nmoves/2.0)

        # Generate samples from posterior:
        if sample_posterior:
            self.sample_posterior(nwalkers=nwalkers, nmoves=nmoves)
            samples = self._flatten_samples(burnin=burnin)
        else:
            samples = None

        biases = self._gen_biases(samples)

        return biases