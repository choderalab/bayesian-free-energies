import numpy as np
from bams.example_systems import GaussianMixtureSampler, IndependentMultinomialSamper, ArgonTemperingSampler

"""
Tests for mixture sampling examples. These example samplers are used to test and demonstrate Bayesian adaptive mixture
sampling (BAMS) as well as self adjusted mixture sampling (SAMS).

Author: Gregory Ross
"""

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
        Ensures the current state is correctly reported.
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

class TestArgonTemperingSampler(object):
    """
    A set of tests for the simulated tempering example of an argon gas.
    """
    def test_initialization(self):
        """
        Ensure initialization works.
        """
        nparticles = 1000
        temperature_ladder = np.linspace(300.0, 500.0, 20)
        biases = np.arange(len(temperature_ladder))
        sampler = ArgonTemperingSampler(nparticles, temperature_ladder, biases)

    def test_sampler(self):
        """
        Test whether object can sample configurations without error
        """
        sampler = ArgonTemperingSampler(100, np.linspace(300.0, 400.0, 20))
        sampler.sample(nsteps=10, niterations=5, save_freq=1)

    def test_current_state(self):
        """
        Ensure that current_state is an array where the only non-zero element is the sampler state.
        """
        sampler = ArgonTemperingSampler(100, np.linspace(300.0, 400.0, 20))
        current_state = sampler.sample(nsteps=1, niterations=1, save_freq=1)
        assert np.where(current_state == 1)[0] == sampler.state

    def test_histogram_counts(self):
        """
        Make sure the histogram correctly tracks the total number of states visited.
        """
        niterations = 5
        sampler = ArgonTemperingSampler(100, np.linspace(300.0, 400.0, 20))
        sampler.sample(nsteps=1, niterations=niterations, save_freq=1)
        assert np.sum(sampler.histogram) == niterations

    def test_sample_state_coverage(self):
        """
        Make sure that when the temperature ladder is all the same temperature and there no biases are applied
        a number of states are visited. Tests that states can be sampled over.

        This is a stochastic test. However, the probability for remaining in same state for 100 iterations when there
        are 5 total number of states with uniform wieghts is roughly 1E-70.
        """
        nstates = 5
        sampler = ArgonTemperingSampler(100, np.repeat(300.0, nstates))
        sampler.sample(nsteps=1, niterations=100, save_freq=1)
        assert np.sum(sampler.histogram > 1) == len(sampler.histogram)

    def test_reset_statistics(self):
        """
        Make sure the sampling statistics are correctly reset
        """
        sampler = ArgonTemperingSampler(100, np.linspace(300.0, 400.0, 20))
        sampler.sample(nsteps=1, niterations=5, save_freq=1)
        sampler.reset_statistics()
        assert np.sum(sampler.histogram) == 0 and sampler.nmoves == 0

    def test_reduced_potential_scaling(self):
        """
        Make sure that the reduced potential has the correct scaling with temperature. The reduced potential has the
        form


            u(x) = E(x)/kT,

        where E(x) is the potential energy, k is Boltzmann's constant and T is the temperature. As T -> infinity,
        u(x)-> 0. This test ensures this correct scaling.
        """
        sampler = ArgonTemperingSampler(100, np.linspace(300.0, 400.0, 20))
        u = sampler.reduced_potential()
        assert np.all(np.diff(u) < 0)

