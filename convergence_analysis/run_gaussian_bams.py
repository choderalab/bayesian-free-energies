import numpy as np
from bams.convergence_analysis_tools import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record the mean squared error for multiple BAMS runs in system of"
                                                 " optimally spaced Gaussian distributions.")
    parser.add_argument('-o', '--out', type=str, help="the name of the file that will store the biases",
                        default="out")
    parser.add_argument('-r', '--repeats', type=int,
                        help="the number times a target is drawn and adaptive inference is performed", default=50)
    parser.add_argument('-c', '--cycles', type=int,
                        help="the number of cycles of state sampling and inferece", default=1000)
    parser.add_argument('-m', '--method', type=str, help="the method used to select the sampling biases",
                        choices=['thompson', 'map', 'mean', 'median'], default='map')
    parser.add_argument('--p_spread', type=float,
                        help='The standard deviation of prior on the free energies', default=10.0)

    args = parser.parse_args()

    # Generate the distributions
    s_min = 1
    s_max = 500
    nstates = 10
    sigmas_optimal = gen_optimal_sigmas(s_min, s_max, nstates)

    aggregate_mse = np.zeros((args.repeats, args.cycles))
    for r in range(args.repeats):
        aggregate_mse[r,:] = bayes_mse_gaussian(sigmas_optimal, args.cycles, nmoves=1, save_freq=1, method=args.method)
        np.save(file=args.out, arr=np.array(aggregate_mse))
