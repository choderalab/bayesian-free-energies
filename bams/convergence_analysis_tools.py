import numpy as np
from copy import deepcopy
from bams.testsystems import *
from bams.bayes_adaptor import BayesAdaptor
from bams.sams_adapter import SAMSAdaptor

#---Functions to compute SAMS and BAMS mean-squared error using the `GaussianMixtureSampler`---#

def gaussian_thermodynamic_length(s_min, s_max):
    """
    Function to compute the thermodynamic length between one-dimensional Gaussian distributions that differ
    only by their standard deviation.
    """
    return np.sqrt(2.0) * np.log(s_max / s_min)

def gen_optimal_sigmas(s_min, s_max, N):
    """
    Generate N standard deviations between s_min and s_max (inclusive) such that the resultant Gaussian distributions
    are equally spaced with respect to thermodynamic length.
    """
    sigmas = np.repeat(s_min, N)
    n = np.arange(0, N)
    return sigmas * np.exp((n * gaussian_thermodynamic_length(s_min, s_max)) / (np.sqrt(2) * (N - 1)))

def gen_sigmas(sigma1, f):
    """
    Generate standard deviations for one-dimensional Gaussian distributions by the relative free energy of the
    normalizing constants.

    Parameters
    ----------
    sigma1: float
        the standard deviation from which all other standard deviations are calculated relative to
    f: numpy.ndarray or float
        the relative free energy of the other Gaussian distributions to sigma1

    Returns
    -------
    numpy.ndarray
        vector of standard deviations
    """
    return sigma1 * np.exp(-f)

def rb_mse_gaussian(sigmas, niterations, nmoves=1, save_freq=1, beta=0.6, flat_hist=0.2):
    """
    Function to compute the mean-squared error from the Rao-Blackwellized update scheme in when sampling
    states with GaussianMixtureSampler.

    Parameters
    ----------
    sigmas: numpy array
        The standard deviations of the Gaussians that are centered on zero
    niterations: int
        The number of iterations of mixture sampling and SAMS updates that will be performed
    nmoves: int
        The number of moves from the Gaussian Gibbs sampler. One position sample and state sample
        constitute one move.
    save_freq: int
        The frequency with which to save the state in the Gaussian mixture sampler, used to decorrelate trajectory
    beta: float
        The exponent in the burn-in phase of the two stage SAMS update scheme.
    flat_hist: float
        The average fractional difference between the target weights of the mixture and count frequency

    Returns
    -------
    mse: numpy array
        the mean-squared error of the SAMS estimate for each iteration
    """
    generator = GaussianMixtureSampler(sigmas=sigmas)
    nstates = len(sigmas)
    adaptor = SAMSAdaptor(nstates=nstates, beta=beta, flat_hist=flat_hist)

    # The target free energy
    f_true = -np.log(sigmas)
    f_true = f_true - f_true[0]

    mse = np.zeros((niterations))
    for i in range(niterations):
        generator.sample(nmoves, save_freq)
        state = generator.state
        noisy = generator.weights
        z = -adaptor.update(state=state, noisy_observation=noisy, histogram=generator.histogram)
        generator.zetas = z
        mse[i] = np.mean((f_true[1:] - z[1:]) ** 2)
    return mse

def binary_mse_gaussian(sigmas, niterations, nmoves=1, save_freq=1, beta=0.6, flat_hist=0.2):
    """
    Function to compute the mean-squared error from the binary update scheme in when sampling
    states with GaussianMixtureSampler.

    Parameters
    ----------
    sigmas: numpy array
        The standard deviations of the Gaussians that are centered on zero
    niterations: int
        The number of iterations of mixture sampling and SAMS updates that will be performed
    nmoves: int
        The number of moves from the Gaussian Gibbs sampler. One position sample and state sample
        constitute one move.
    save_freq: int
        The frequency with which to save the state in the Gaussian mixture sampler, used to decorrelate trajectory
    beta: float
        The exponent in the burn-in phase of the two stage SAMS update scheme.
    flat_hist: float
        The average fractional difference between the target weights of the mixture and count frequency

    Returns
    -------
    mse: numpy array
        the mean-squared error of the SAMS estimate for each iteration
    """
    generator = GaussianMixtureSampler(sigmas=sigmas)
    nstates = len(sigmas)
    adaptor = SAMSAdaptor(nstates=nstates, beta=beta, flat_hist=flat_hist)

    # The target free energy
    f_true = -np.log(sigmas)
    f_true = f_true - f_true[0]

    mse = np.zeros((niterations))
    for i in range(niterations):
        noisy = generator.sample(nmoves, save_freq)
        state = generator.state
        z = -adaptor.update(state=state, noisy_observation=noisy, histogram=generator.histogram)
        generator.zetas = z
        mse[i] = np.mean((f_true[1:] - z[1:]) ** 2)
    return mse

def bayes_mse_gaussian(sigmas, niterations, nmoves=1, save_freq=1, method='thompson', enwalkers=30, enmoves=100):
    """
    Function to compute the mean-squared error from a Bayesian update scheme when sampling
    states with GaussianMixtureSampler.

    Parameters
    ----------
    sigmas: numpy array
        The standard deviations of the Gaussians that are centered on zero
    niterations: int
        The number of iterations of mixture sampling and SAMS updates that will be performed
    nmoves: int
        The number of moves from the Gaussian Gibbs sampler. One position sample and state sample
        constitute one move.
    save_freq: int
        The frequency with which to save the state in the Gaussian mixture sampler, used to decorrelate trajectory
    method: string
        the method with which new samples are generated. Either 'map', 'thompson', 'mean', or 'median'.

    Returns
    -------
    mse: numpy array
        the mean-squared error of the Bayesian estimate for each iteration
    """
    # Initialize the sampler
    generator = GaussianMixtureSampler(sigmas=sigmas)
    generator.sample(nmoves, save_freq)
    histogram = generator.histogram
    generator.reset_statistics()
    # Intialize the Bayesian adaptor
    adaptor = BayesAdaptor(generator.zetas, histogram, method=method)

    # The target free energy
    f_true = -np.log(sigmas)
    f_true = f_true - f_true[0]

    mse = np.zeros((niterations))
    for i in range(niterations):
        # Generate new biases
        z = adaptor.update(nwalkers=enwalkers, nmoves=enmoves)
        generator.zetas = z
        # Sample from Gaussian mixture, recording the histogram
        generator.sample(nmoves, save_freq)
        h = deepcopy(generator.histogram)
        generator.reset_statistics()
        # Keep track of new counts and biases
        adaptor.counts = np.vstack((adaptor.counts, h))
        adaptor.zetas = np.vstack((adaptor.zetas, z))

        # Calculate the error
        estimate = adaptor.map_estimator()
        mse[i] = np.mean((f_true[1:] - estimate) ** 2)

    return mse

#---Functions to compute SAMS and BAMS mean-squared error using the `IndependentMultinomialSamper`---#

def multinomial_sams_error(repeats, niterations, f_range, beta, nstates=2):
    """
    Function to compute the mean-squared error of the SAMS binary update scheme when drawing samples from
    the multinomial distribution. Over many repeats, target free energies will be drawn randomly and uniformly
    from a specified range and SAMS will adapt to those free energies.

    Parameters
    ----------
    repeats: int
        The number of repeats with which to draw target free energies and run SAMS
    ninterations:
        The number of state samples generated and SAMS adaptive steps
    f_range: float
        The interval over which target free energies will be drawn
    beta: float
        The exponent for the SAMS burn-in stage
    nstates: int
        The number of states and target free energies.
    """
    binary_aggregate_msd = np.zeros((repeats, niterations))
    for r in range(repeats):
        f_true = np.random.uniform(low=-f_range / 2.0, high=f_range / 2.0, size=nstates)
        f_true -= f_true[0]
        sigmas = gen_sigmas(sigma1=1, f=f_true)

        generator = IndependentMultinomialSamper(free_energies=f_true)
        adaptor = SAMSAdaptor(nstates=nstates, beta=beta)

        for i in range(niterations):
            noisy = generator.sample()
            state = np.where(noisy != 0)[0][0]
            z = -adaptor.update(state=state, noisy_observation=noisy, histogram=generator.histogram)
            generator.zetas = z
            binary_aggregate_msd[r, i] = np.mean((f_true - z) ** 2)
    return binary_aggregate_msd



def run_bams_example(prior, spread, location, f_true, method, logistic=False, ncycles=50, nsamps=1, enmoves=200,
                     enwalkers=50):
    """
    Function to estimate the bias and variance of the BAMS method as a function of iteration

    Parameters
    ----------
    prior = str
        The type of prior used, either 'gaussian', 'laplace', or 'cauchy'
    spread = numpy array
        The value of spread parameter for the prior, e.g the standard deviation for the Gaussian prior
    location = numpy array
        The location parameter for the prior, e.g the mode for the Laplace prior
    method = string
        The method used in the update procedure, either 'thompson' or 'map'
    logistic = bool
        Whether to convolute the bias generation procedure with the logisitic distribution
    ncycles = int
        The number of iterations for state sampling and adaptive estimation
    nsamps = int
        The number of state samples generated per cycle
    enmoves = int
        The number of emcee moves performed for each walker
    enwalkers = int
        The number emcee walkers

    Returns
    -------
    bias: numpy array
        The mean-squared distance between the MAP estimate and target free energy for each iteration
    variance: numpy array
        The variance of the posterior distribution at each stage of the iteration

    """
    #TODO: Fix this function to relect most recent refactoring of BayesAdaptor
    # Generating the true state probabilities:
    p = np.hstack((1, np.exp(-f_true)))
    p = p / np.sum(p)
    # Pre-assigment
    map_estimate = []
    zetas = [np.repeat(0, len(f_true) + 1)]  # Starting the initial bias at zero
    counts = []
    variance = []
    # Online estimation of free energies:
    for i in range(ncycles):
        # Sample from multinomial
        q = p * np.exp(zetas[-1])
        q = q / np.sum(q)
        counts.append(np.random.multinomial(nsamps, q))
        # Sample from the posterior
        adaptor = BayesAdaptor(zetas=np.array(zetas), counts=np.array(counts), prior=prior, spread=spread,
                               location=location)
        adaptor.sample_posterior(nwalkers=enwalkers, nmoves=enmoves)
        # Sample a new biasing potential
        zetas.append(np.hstack((0.0, adaptor.gen_biases(method=method, logistic=logistic))))
        # Collect data
        f_guess = np.hstack((0.0, adaptor.flat_samples.mean(axis=0)))
        map_estimate.append(adaptor.map_estimator(f_guess=f_guess))
        variance.append(np.var(adaptor.flat_samples))
    # Calculate the bias
    map_estimate = np.array(map_estimate)
    bias = (map_estimate - f_true) ** 2
    variance = np.array(variance)
    return bias, variance
