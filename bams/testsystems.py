import numpy as np

class IndependentMultinomialSamper(object):
    """
    Class to draw independent samples from a biased multinomial distribution with the following distribution:

        p_i = exp(zeta_i - f_i) / sum_i[exp(zeta_i - f_i)]

    where f_i and zeta_i are the free energy and applied bias the free energy of ith state, respectively. The unbiased
    distribution has propabilities proportional to exp(-f_i).
    """
    def __init__(self, free_energies=None, zetas=None):
        """
        Initialize the biased multinomial sampler.

        Parameters
        ----------
        free_energies: numpy array
            the free energy of the unbiased probabilities
        zetas: numpy array
            the exponent of the bias applied to the probabilities
        """
        if free_energies is None:
            self.free_energies = np.random.uniform(-50, 50, size=5)
            self.free_energies -= self.free_energies[0]
        else:
            self.free_energies = free_energies
            self.free_energies -= self.free_energies[0]

        if zetas is not None:
            self.zetas = zetas
            self.zetas -= self.zetas[0]
        else:
            self.zetas = np.repeat(0.0, len(self.free_energies))

        self.state_counter = np.repeat(0.0, len(self.free_energies))

    def sample(self, nsamples=1):
        """
        Sample from multinomial distribution and update the state histogram

        Parameters
        ----------
        nsamples: int
            the number of samples to draw

        Returns
        -------
        current_state: numpy array
            binary array where the only non-zero element indicates the current state of the system
        """
        p = np.exp(self.zetas - self.free_energies)
        p = p / np.sum(p)

        self.state_counter += np.random.multinomial(nsamples - 1, p)
        current_state = np.random.multinomial(1, p)
        self.state_counter += current_state

        return current_state

    def reset_statistics(self):
        """
        Reset the histogram state counter to zero
        """
        self.state_counter -= self.state_counter

class GaussianMixtureSampler(object):
    """
    Class to sample positions and states of a one-dimensional mixture of Gaussian distributions with a
    different standard deviations. The mixture is of the form

        p_i(x) = q_i(x) * exp(zeta_i) / sum_i[Z_i * exp(zeta_i)],

    where x is the position, q_i is the unnormalized density of the ith Gaussian, Z_i is normalizing
    constant of the ith distribution and zeta_i is the exponentiated weight of the ith element. All Gaussians are
    centered on zero.

    Example
    -------
    Initialize the standard deviations of the Gaussians that will make up the mixture
    >>> sigmas = np.array((1.0, 10.0, 20.0, 30.0))

    Specify the weights of the mixture.
    >>> weights = np.array((1, 3, 10, 4))

    The weights don't need to be normalized, but the class expects the exponential of the weights i.e.
    >>> zetas = np.exp(weights)

    Initialize the sampler
    >>> sampler = GaussianMixtureSampler(sigmas=sigmas, zetas=zetas)

    Sample positions and states (where state is a one of the Gaussians in the mixture) for 500 steps
    and record the state every 10 steps.
    >>> sampler.sampler_mixture(initerations=500, save_freq=10):

    One iteration corresponds to an independence sample from the current state and a Gibbs sample of the state.
    current. View the number of times a visit to a state was recorded
    >>> print sampler.state_counter

    """
    def __init__(self, sigmas=np.arange(1,100,10), zetas=None):
        """
        Initialisation of the multiple states of the Gaussian mixture. The weight of each Gaussian in the mixture
        is specified by the negative exponential of the biasing potential.

        Parameters
        ----------
        sigmas: numpy.ndarray
          the standard deviations of each state/gaussian
        zeta: numpy.ndarray
          the biasing potential for each state

        Returns
        -------

        """
        self.sigmas = np.array(sigmas)
        if zetas is not None:
            if np.array(zetas).shape != self.sigmas.shape:
                raise Exception('The number of biasing potentials (zetas) must match the length of sigmas')
            else:
                self.zetas = np.array(zetas)
        else:
            self.zetas = np.zeros(len(sigmas))

        self.position = 1.0   # The position of the oscillator

        # Pick the current state
        self.state = np.random.choice(len(self.sigmas))

        # Pre-assign the conditional probability of the state given the current position. Updated with self._sample_state.
        self.weights = np.zeros(len(self.sigmas))

        # Initialise tracking statistics
        self.state_counter = np.zeros(len(self.sigmas))
        self.nmoves = 0

    def _get_energy(self):
        """
        The negative log of the probability density of the position for each state

        Returns
        -------
        energy: numpy.array
          the energy of the oscillator for each sigma
        """
        return (self.position**2.0)/2.0/(self.sigmas**2)

    def _sample_state(self):
        """
        Sample the state conditioned on the position and bias

        """
        q = np.exp(-self._get_energy() + self.zetas)
        weights = q / np.sum(q)
        self.weights = weights
        states = range(len(self.sigmas))
        self.state = np.random.choice(states, p=weights)

    def _sample_position(self):
        """
        Sample the oscillator from a normal distribution centered on zero with a sigma given by the current state

        """
        self.position = np.random.normal(loc=0, scale = self.sigmas[self.state])

    def sample(self, niterations=500, save_freq=50):
        """
        Iterate through alternating state and position sampling and save the state at a specified rate.

        Parameters
        ----------
        niterations: int
          the number of cycles of position and state sampling
        save_freq: int
          the state will be recorded at every multiple of this number

        Return
        ------
        current_state: numpy array
            binary array where the only non-zero element indicates the current state of the system
        """
        for iteration in range(1, niterations+1):
            self._sample_position()
            self._sample_state()
            if iteration % save_freq == 0:
                self.nmoves += 1
                self.state_counter[self.state] += 1

        current_state = np.zeros(len(self.sigmas))
        current_state[self.state] = 1
        return current_state

    def reset_statistics(self):
        """
        Reset the state histogram and move counter
        """
        self.state_counter = np.zeros(len(self.sigmas))
        self.nmoves = 0
