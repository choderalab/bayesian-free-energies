import numpy as np
from bams.bayes_adaptive_fitter import BayesAdaptor

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record the bias and variance for multiple BAMS runs in a two-state"
                                                 " system with independent sampling.")
    parser.add_argument('-b', '--bias', type=str, help="the name of the file that will store the biases",
                        default="bias")
    parser.add_argument('-v', '--var', type=str,
                        help="the name of the file that will store the biases",
                        default="variance")
    parser.add_argument('-r', '--repeats', type=int,
                        help="the number times a target is drawn and adaptive inference is performed", default=500)
    parser.add_argument('-c', '--cycles', type=int,
                        help="the number of cycles of state sampling and inferece", default=50)
    parser.add_argument('-m', '--method', type=str, help="the method used to select the sampling biases",
                        choices=['thompson', 'map', 'mean', 'median'], default='map')
    parser.add_argument('--t_spread', type=float,
                        help='The minimum and maximum values from which the target free energies are drawn from', default=200.0)
    parser.add_argument('--p_spread', type=float,
                        help='The standard deviation of prior on the free energies', default=100.0)

    args = parser.parse_args()

    # Running BAMS over many repeats
    bias = []
    variance = []
    for r in range(args.repeats):
        f_true = -np.random.uniform(-args.t_spread,args.t_spread,size=1)
        b, v = run_bams_example('gaussian', spread=args.p_spread, location=0.0, f_true=f_true, method=args.method,
                                ncycles=args.cycles, nsamps=1)
        bias.append(b)
        variance.append(v)
        np.save(file=args.bias, arr=np.array(bias))
        np.save(file=args.var, arr=np.array(variance))
