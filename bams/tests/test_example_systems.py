import numpy as np
from bams.example_systems import GaussianMixtureSampler, IndependentMultinomialSamper


class TestGaussianSampler(object):
    """
    Tests for the Gaussian mixture sampler defined in bams.example_systems.
    """
    def test_sampler(self):
        """
        Run the sampler with without biases
        """
        # Create 10 states with different standard deviations.
        sampler = GaussianMixtureSampler(sigmas=np.arange(1, 100, 10))

        # Perform the mixture sampling
        sampler.step(nsteps=1, save_freq=1)

    def test_biased_sampler(self):
        """
        Run the sampler with after the application of biases
        """
        # Create 10 states with different standard deviations.
        sampler = GaussianMixtureSampler(sigmas=np.arange(1, 100, 10), zetas=-np.arange(1, 100, 10))

        # Perform the biased mixture sampling
        sampler.step(nsteps=1, save_freq=1)

    def test_current_state(self):
        """
        Ensures the current state agrees with the histogram.
        """
        # Create 10 states with different standard deviations.
        sampler = GaussianMixtureSampler(sigmas=np.arange(1, 100, 10))

        # Perform the mixture sampling with a single move
        sampler.step(nsteps=1, save_freq=1)

        # See if the current state corresponds to the non-zero element in the histogram
        assert sampler.state == int(np.where(sampler.histogram != 0)[0][0])

    def test_histogram_counts(self):
        """
        Ensures that after several samples, the histogram counts are consistent with the number of number of samples
        """
        # Create 10 states with different standard deviations.
        sampler = GaussianMixtureSampler(sigmas = np.arange(1,100,10))

        # Perform the mixture sampling
        nsteps = 10
        sampler.step(nsteps=nsteps, save_freq=1)

        # Ensure the number of iterations matches the counts
        assert sampler.histogram.sum() == nsteps

class TestMultinomialSampler(object):
    """
    Contains tests for the multinomial sampler in bams.IndependentMultinomialSamper
    """
    def test_sampler(self):
        """
        Generates samples from a multinomial distribution
        """
        # Initialize sampler with example free energies
        free_energies = np.arange(0,-100, -10)
        sampler = IndependentMultinomialSamper(free_energies=free_energies)

        # Run the sampler
        sampler.step(nsteps=10)

    def test_biased_sampler(self):
        """
        Generates samples from a multinomial distribution
        """
        # Initialize sampler with example free energies
        free_energies = np.arange(0,-100, -10)
        sampler = IndependentMultinomialSamper(free_energies=free_energies, zetas=-np.arange(1, 100, 10))

        # Run the sampler
        sampler.step(nsteps=10)

    def test_current_state(self):
        """
        Ensures the current state agrees with the histogram.
        """
        # Initialize sampler with example free energies
        free_energies = np.arange(0,-100, -10)
        sampler = IndependentMultinomialSamper(free_energies=free_energies)

        # Perform the mixture sampling with a single move
        sampler.step(nsteps=1)

        # See if the current state corresponds to the non-zero element in the histogram
        assert sampler.state == int(np.where(sampler.histogram != 0)[0][0])

    def test_histogram_counts(self):
        """
        Ensures that after several samples, the histogram counts are consistent with the number of number of samples
        """
        # Initialize sampler with example free energies
        free_energies = np.arange(0,-100, -10)
        sampler = IndependentMultinomialSamper(free_energies=free_energies)

        # Run the sampler
        nsteps = 10
        sampler.step(nsteps=nsteps)

        assert sampler.histogram.sum() == nsteps

