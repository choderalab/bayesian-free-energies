import numpy as np
import bams.convergence_analysis_tools as ctools

class TestConvergenceAnalyisTools(object):
    """
    Suit of tests for the functions used to assess the error and empirical convergence properties of BAMS and SAMS.
    The test are extremely basic as they only ensure that each function throws no errors.
    """
    def test_gaussian_thermolength(selfs):
        """
        Ensure function to calculate the thermodynamic length between two Gaussians that differ only in their standard
        deviation throws no errors
        """
        ctools.gaussian_thermodynamic_length(s_min=1, s_max=100)

    def test_generate_standard_deviations(self):
        """
        Ensure function to generate standard deviations for 1D Gaussians evenly across the thermodynamic length works
        """
        ctools.gen_optimal_sigmas(s_min=1, s_max=100, N=10)

    def test_rb_gaussian_mse(self):
        """
        Ensure the function to compute the mean-squared error per iteration works for Rao-Blackwellized SAMS when
        sampling from a mixture of Gaussians.
        """
        sigmas = np.arange(1, 4)
        ctools.rb_mse_gaussian(sigmas=sigmas, niterations=1, nmoves=1, save_freq=1, beta=0.6, flat_hist=0.2)

    def test_binary_gaussian_mse(self):
        """
        Ensure the function to compute the mean-squared error per iteration works for SAMS binary update scheme when
        sampling from a mixture of Gaussians.
        """
        sigmas = np.arange(1, 4)
        ctools.binary_mse_gaussian(sigmas=sigmas, niterations=1, nmoves=1, save_freq=1, beta=0.6, flat_hist=0.2)

    def test_bayes_gaussian_mse(self):
        """
        Ensure the function to compute the mean-squared error per iteration works for BAMS when sampling from a
        mixture of Gaussians.
        """
        sigmas = np.arange(1, 4)
        ctools.bayes_mse_gaussian(sigmas=sigmas, niterations=1, nmoves=1, save_freq=1, method='thompson', enwalkers=8,
                                  enmoves=1)

    def test_binary_multinomial_mse(self):
        """
        Ensure the basic functionality of the function used to compute the mean-squared error of the SAMS binary update
        method on indpendent samples drawn from a multinomial distribution.
        """
        ctools.binary_mse_multinomial(repeats=1, niterations=1, f_range=1, beta=0.6, nstates=2)

    def test_bayes_multinomial_mse(self):
        """
        Ensure the basic functionality of the function used to compute the mean-squared error of the Bayesian update
        methods on indpendent samples drawn from a multinomial distribution.
        """
        f_true = np.array((0, 100))
        ctools.bayes_mse_multinomial(niterations=1, prior='gaussian', spread=100.0, location=0.0, f_true=f_true,
                                     method='thompson', enmoves=1, enwalkers=4)
