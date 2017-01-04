import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd import grad
from scipy import optimize
import emcee
from copy import deepcopy

class MultinomialBayes(object):
    """
    Class to estimate free energies from multinomial samples via Bayesian estimation
    """
    def __init__(self, zetas, counts, free_energies=None, prior='gaussian', location=None, spread=None):
        """
        Parameters
        ----------
        zetas: numpy.ndarray
            the biasing potentials applied to each state, where the columns correspond to the states and rows repeats.
        counts: numpy.ndarray
            the number of times the state was visited for a number of repeats, where the columns correspond to the states
            and rows repeats.
        free_energies: numpy.ndarray
            the initial guess of the free energies for each state, relative to the first
        prior: string
            the name of the prior distribution. Choice is between 'gaussian', 'laplace', and 'cauchy'
        location: float, numpy.ndarray
            the location parameter of the prior for each free energy, e.g. the mean for 'gaussian'. If float, the same
            value is applied to all free energies.
        spread: float, numpy.ndarray
            the spread parameter of the prior for each free energy, e.g. the standard deviation for 'gaussian'.
            If float, the same value is applied to all free energies.
        """

        if zetas.shape != counts.shape:
            raise Exception('Error: the dimensions of the biasing potentials and state counts must match')

        # Formatting all input data into a matrix with the number of rows equals the number of repeats
        if zetas.ndim == 1:
            zetas = np.array([zetas])
            counts = np.array([counts])

        self.zetas = zetas #- zetas[0, 0]
        self.counts = counts

        # If no initial guess of the free energies is supplied, set the free energies to the relative counts
        if free_energies is None:
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

    def _sum_of_squares(self, f):
        """
        The sum of squares for the predicted state counts for a given estimate of the relative free energies of the
        states.

        Parameters
        ----------
        f:numpy.ndarray
            Estimate of the free energies for each state relative to the first (f[0])

        Returns
        -------
        float
            The sum of squares
        """
        prediction = self._expected_counts(f)
        return np.sum((self.counts - prediction)**2)

    def fit_least_squares(self, f_guess=None, max_iter=1000, precision=0.00001):
        """
        Predict the free energies of each state by minimizing the sum of squares of the observed and predicted counts.

        Note
        ----
        This function uses `autograd`.

        Returns
        -------
        free

        """

        if f_guess is None:
            f_guess = deepcopy(self.free_energies)

        def line_search(f, gradient, t=1.0, a=0.5):
            """
            Backtracking line search algorithm
            """
            loss = self._sum_of_squares(f)
            while self._sum_of_squares(f - t * gradient) >= loss + a * t * np.sum(gradient**2) / 2:
                t *= a
            return t

        # Creating a gradient function with autograd
        training_grad = grad(self._sum_of_squares)

        # Fit free energies by gradient decent
        i = 1
        step_max = precision * 100
        while (i <= max_iter) and step_max >= precision:
            g = training_grad(f_guess)
            t = line_search(f_guess,g)
            step = t*g[1:]
            f_guess[1:] -= step
            step_max = step.max()
            i += 1
            f_guess -= f_guess[0]
        return f_guess

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

    def _log_likelihood(self, f):
        """
        The log likelihood of the counts, without terms proportional to the free energy.
        The normalisation constant discarded.

        Parameter
        ---------
        f: numpy.ndarray
          the vector of estimates for the free energy

        Returns
        -------
        l: float
          the log of the unnormalized likelihood
        """
        rn = np.sum(self.counts, axis=0)    # Sum of the counts across the repeats at each zeta index
        zn = np.sum(self.counts, axis=1)    # Sum of the counts across the zetas at each repeat
        l = -np.sum(rn * f) - np.sum(zn * logsumexp(self.zetas - f, axis=1))

        return l

    def _line_search(self, loss, f, gradient, t=1.0, a=0.5):
        """
        Backtracking line search algorithm used in self.max_a_post. Determines a suitable step size to take during
        gradient decent.

        Parameters
        ----------
        loss: function
            the loss function to be minimized
        f: numpy.ndarray
            the vector of free energies
        gradient: numpy.ndarray
            the vector gradient (grad) of the log posterior at f
        t: float
            initial step size for gradient decent
        a: float, should be 0 < a < 1
            factor that t decreases decreases by at each iteration

        Returns
        -------
        t: float
            final step size to be used in gradient decent
        """
        #TODO: make it work better on multidimentional functions?
        while loss(f - t * gradient) >= loss(f) + a * t * np.sum(gradient**2) / 2:
            t *= a
        return t

    def _deprecated_map_estimator(self, f_guess=None, max_iter=5000, precision=0.0001):
        """
        An old version of a maximum a posteriori estimator for the free energies, relative to state 1. This uses a
        custom gradient decent algorithm.

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
            f_guess = deepcopy(self.free_energies)

        # Defining an internal loss function to minimize for fitting.
        if self.prior == 'gaussian':
            def loss(f):
                """
                The negative log of the posterior with Gaussian priors on the free energies
                """
                return -self._log_likelihood(f) - self._log_prior_gaussian(f)
        elif self.prior == 'laplace':
            def loss(f):
                """
                The negative log of the posterior with Laplace priors on the free energies
                """
                return -self._log_likelihood(f) - self._log_prior_laplace(f)
        elif self.prior == 'cauchy':
            def loss(f):
                """
                The negative log of the posterior with Cauchy priors on the free energies
                """
                return -self._log_likelihood(f) - self._log_prior_cauchy(f)
        else:
            raise Exception('The prior "{0}" is not supported'.format(self.prior))

        # Creating a gradient function with autograd
        training_grad = grad(loss)

        # Fit free energies by gradient decent
        i = 1
        step_max = precision * 100
        while (i <= max_iter) and step_max >= precision:
            g = training_grad(f_guess) # The negative of the gradient as maximizing the posterior
            t = self._line_search(loss, f_guess, g)
            step = t*g[1:]
            f_guess[1:] -= step
            step_max = step.max()
            i += 1
            f_guess -= f_guess[0]

        return f_guess

    def map_estimator(self, f_guess=None, method='BFGS'):
        """
        Provides a maximum a posteriori estimate of the free energies, relative to state 1. 

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

        return np.hstack((0.0, fit.x))

    def sample_posterior(self, nwalkers = 50, nmoves = 500, f_guess=None, ):
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
            #f_guess = self.max_a_post()
            f_guess = self.free_energies

        # The number of free parameters is len(f_guess - 1), as free energies will be relative to first.
        initial_positions = [f_guess[1:] + 1e-1*np.random.randn(len(f_guess) - 1) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, len(f_guess) - 1, log_posterior)
        sampler.run_mcmc(initial_positions, nmoves)

        return sampler.chain