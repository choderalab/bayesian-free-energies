import numpy as np
from harmonic_mixture_sampler import GaussianMixtureSwapper


def test_gaussian_mixture_sampler():
    """
    Run the Gaussian mixture sampler, and ensure the counts are consistent with the number of samples
    """
    # Create 10 states with different standard deviations.
    swapper = GaussianMixtureSwapper(sigmas = np.arange(1,100,10))

    # Perform the mixture sampling
    niterations = 100
    swapper.sample_mixture(niterations=niterations, save_freq=1)

    # Ensure the number of iterations matches the counts
    assert swapper.state_counter.sum() == niterations