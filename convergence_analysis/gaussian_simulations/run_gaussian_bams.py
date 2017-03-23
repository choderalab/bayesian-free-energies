import numpy as np
import bams.convergence_analysis_tools as ctools

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record the mean squared error for multiple BAMS runs in system of"
                                                 " optimally spaced Gaussian distributions.")
    parser.add_argument('-o', '--out', type=str, help="the name of the file that will store the error as a function of"
                                                      "time", default="out")
    parser.add_argument('-r', '--repeats', type=int,
                        help="the number times a target is drawn and adaptive inference is performed", default=50)
    parser.add_argument('-c', '--cycles', type=int,
                        help="the number of cycles of state sampling and inferece, default=1000", default=1000)
    parser.add_argument('-m', '--method', type=str, help="the method used to select the sampling biases",
                        choices=['thompson', 'map', 'mean', 'median'], default='map')
    parser.add_argument('--smin', type=float,
                        help='The lowest standard deviation of the mixture components', default=1.0)
    parser.add_argument('--smax', type=float,
                        help='The highest standard deviation of the mixture components', default=500.0)
    parser.add_argument('--nstates', type=int,
                        help='The number of compenents in the mixture distribution', default=10)
    parser.add_argument('--p_spread', type=float,
                        help='The standard deviation of prior on the free energies, default=10.0', default=10.0)
    parser.add_argument('--nmoves', type=int,
                        help='The number of steps of the Gibbs sampler per iteration', default=1)
    parser.add_argument('--save_freq', type=int,
                        help='The frequency at which states are recorded in the Gibbs sampler', default=1)

    args = parser.parse_args()

    # Generate the distributions by equally spacing in terms of thermodynamic length
    sigmas_optimal = ctools.gen_optimal_sigmas(args.smin,args.smax, args.nstates)

    aggregate_mse = np.zeros((args.repeats, args.cycles))
    for r in range(args.repeats):
        aggregate_mse[r,:] = ctools.bayes_mse_gaussian(sigmas_optimal, args.cycles, nmoves=args.nmoves,
                                                       save_freq=args.save_freq, method=args.method)
        np.save(file=args.out, arr=np.array(aggregate_mse))
