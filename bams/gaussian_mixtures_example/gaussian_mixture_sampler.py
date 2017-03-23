import numpy as np

class GaussianMixtureSampler(object):
    """
    Class to step positions and states of a one-dimensional mixture of Gaussian distributions with a
    different standard deviations. The mixture is of the form

        p_i(x) = q_i(x) * exp(zeta_i) / sum_i[Z_i * exp(zeta_i)],

    where x is the position, q_i is the unnormalized density of the ith Gaussian, Z_i is normalizing
    constant of the ith distribution and zeta_i is the exponentiated weight of the ith element.

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

    One iteration corresponds to an independence step from the current state and a Gibbs step of the state.
    current. View the number of times a visit to a state was recorded
    >>> print sampler.state_counter

    """
    def __init__(self, sigmas = np.arange(1,100,10), zetas = None):
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

        # Initialise tracking statistics
        self.state_counter = np.zeros(len(self.sigmas))
        self.nmoves = 0

    def _get_energy(self):
        """
        The negative log of the probability density of the position for each state

        Returns
        -------
        energy: numpy.ndarray
          the energy of the oscillator for each sigma
        """
        return (self.position**2.0)/2.0/(self.sigmas**2)

    def _sample_state(self):
        """
        Sample the state conditioned on the position

        """
        q = np.exp(-self._get_energy() + self.zetas)
        p = q / np.sum(q)
        states = range(len(self.sigmas))
        self.state = np.random.choice(states, p=p)

    def _sample_position(self):
        """
        Sample the oscillator from a normal distribution centered on zero with a sigma given by the current state

        """
        self.position = np.random.normal(loc=0, scale = self.sigmas[self.state])

    def sample_mixture(self, niterations=500, save_freq=50):
        """
        Iterate through alternating state and position sampling and save the state at a specified rate.

        Parameters
        ----------
        niterations: int
          the number of cycles of position and state sampling
        save_freq: int
          the state will be recorded at every multiple of this number

        """
        for iteration in range(1, niterations+1):
            self._sample_position()
            self._sample_state()
            if iteration % save_freq == 0:
                self.nmoves += 1
                self.state_counter[self.state] += 1
