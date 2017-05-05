import numpy as np

class IndependentMultinomialSamper(object):
    """
    Class to draw independent samples from a biased multinomial distribution with the following distribution:

        p_i = exp(zeta_i - f_i) / sum_i[exp(zeta_i - f_i)]

    where f_i and zeta_i are the free energy and applied bias the free energy of ith state, respectively. The unbiased
    distribution has propabilities proportional to exp(-f_i).

    Example
    -------
    Specify the free energy difference between each state
    >>> free_energies = np.array((0.0, -10.0))

    Specify the biases to be applied to each state
    >>> biases = np.array((0.0, 0.0))

    Initialize the sampler
    >>> sampler = IndependentMultinomialSamper(free_energies=free_energies, zetas=biases)

    Take 200 uncorrelated global jumps over the states and record the current location in state_vector
    >>> state_vector = generator.step(nsteps=200)

    The array state_vector is a binary vector with the only non-zero element at the current state of the system. The
    index of the non-zero element is given by
    >>> print(sampler.state)

    View how many times each state was visited over all the steps:
    >>> print(sampler.histogram)
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

        self.histogram = np.repeat(0.0, len(self.free_energies))

        self.state = None

    def step(self, nsteps=1):
        """
        Sample from multinomial distribution and update the state histogram

        Parameters
        ----------
        nsteps: int
            the number of samples to draw

        Returns
        -------
        current_state: numpy array
            binary array where the only non-zero element indicates the current state of the system
        """
        p = np.exp(self.zetas - self.free_energies)
        p = p / np.sum(p)

        self.histogram += np.random.multinomial(nsteps - 1, p)
        current_state = np.random.multinomial(1, p)
        self.histogram += current_state
        self.state = int(np.where(current_state != 0)[0][0])

        return current_state

    def reset_statistics(self):
        """
        Reset the histogram state counter to zero
        """
        self.histogram -= self.histogram

class GaussianMixtureSampler(object):
    """
    Class to step positions and states of a one-dimensional mixture of Gaussian distributions with a
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

    One iteration corresponds to an independence step from the current state and a Gibbs step of the state.
    current. View the number of times a visit to a state was recorded
    >>> print sampler.histogram

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
        self.histogram = np.zeros(len(self.sigmas))
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

    def step(self, nsteps=500, save_freq=50):
        """
        Iterate through alternating state and position sampling and save the state at a specified rate.

        Parameters
        ----------
        nsteps: int
          the number of cycles of position and state sampling
        save_freq: int
          the state will be recorded at every multiple of this number

        Return
        ------
        current_state: numpy array
            binary array where the only non-zero element indicates the current state of the system
        """
        for iteration in range(1, nsteps+1):
            self._sample_position()
            self._sample_state()
            if iteration % save_freq == 0:
                self.nmoves += 1
                self.histogram[self.state] += 1

        current_state = np.zeros(len(self.sigmas))
        current_state[self.state] = 1
        return current_state

    def reset_statistics(self):
        """
        Reset the state histogram and move counter
        """
        self.histogram = np.zeros(len(self.sigmas))
        self.nmoves = 0

class ArgonTemperingSampler(object):
    """
    Class to perform simulated tempering on an Argon gas in the NVT ensemble.

    This class samples from the joint distribution

        p(x,i) = exp(-(f_i + u(x)) / kT_i + zeta_i) / sum_i[exp(-(f_i + u(x)) / kT_i + zeta_i)],

    where u(x) is the energy at configuration x, zeta_i is the bias for the ith state and f_i is the free energy at
    state i:

        exp(-f_i / kT_i) =  \int_X exp(-u(x) / kT_i) dx

    The configuration of system is updated using openmm's default Langevin integrator with a timestep of 2fs.
    The temperature of the system is updated with Gibbs sampling.

    Example
    -------

    This is how to perform an unbiased simulated tempering simulation with 20 states, with temperatures between 300K and
    400K. First, the sampler must be initialized:

    >>> nparticles = 1000  # The number of Argon particles
    >>> temperature_ladder = np.linspace(300.0, 500.0, 20)
    >>> biases = np.zeros(len(temperature_ladder))         # Unbiased simulation means all the biases are zero
    >>> sampler = ArgonTemperingSampler(nparticles, temperature_ladder, biases)

    Running 100 iterations of simulated tempering. Each iteration consists of 2ps of dynamics (nsteps=1000) and the
    state information is stored after every 5 iterations (save_freq=5):

    >>> sampler.sample(nsteps=1000, niterations=100, save_freq=5 )

    View the states that were visited:
    >>> print(sampler.histogram)
    """

    def __init__(self, nparticles, temperature_ladder=np.linspace(300.0, 500.0, 20), biases=None):
        """

        Parameters
        ----------
        nparticles: int
            the number of Argon atoms in the system.
        temperature_ladder: list or array of floats
            the temperatures (in Kelvin) over which tempering will be performed.
        biases: numpy array
            the biasing potentials for each state. If None, then each bias=0.
        """
        # Import the modules that are specific to this class:
        from simtk import openmm, unit
        from openmmtools.testsystems import LennardJonesFluid

        # Initialize the sampler statistics
        self.state = 0
        self.histogram = np.zeros(len(temperature_ladder))
        self.nmoves = 0
        # Pre-assign the conditional probability of the state given the current position.
        self.weights = np.zeros(len(temperature_ladder))

        # Get the biases
        if biases is None:
            self.biases = np.zeros(len(temperature_ladder))
        else:
            self.biases = biases

        # Initialize the temperature and temperature ladder
        self.temperature_ladder = []
        for t in temperature_ladder:
            self.temperature_ladder.append(t * unit.kelvin)
        self.temperature = temperature_ladder[self.state]

        # Boltzmann's constant
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self.kB = kB.in_units_of(unit.kilojoule_per_mole / unit.kelvin)

        # Create the system:
        argon_gas = LennardJonesFluid(nparticles=nparticles, mass=39.9 * unit.amu, sigma=3.4 * unit.angstrom,
                                      epsilon=0.238 * unit.kilocalories_per_mole)
        system, positions = argon_gas.system, argon_gas.positions

        # Initialize the integrator at the first temperature in the list
        self.integrator = openmm.LangevinIntegrator(self.temperature, 1 / unit.picosecond,
                                                    0.002 * unit.picosecond)
        # Create the OpenMM context
        self.context = openmm.Context(system, self.integrator)
        self.context.setPositions(positions)
        self.context.setVelocitiesToTemperature(self.temperature)

    def reduced_potential(self):
        """
        Calculate the reduced potential of the system at every temperature in the ladder. The reduced potential, u_i, is
        given by

            u_i = U / kT_i

        where U is the potential energy, k is Boltzmann's constant, T_i is the ith temperature in the ladder.

        Returns
        -------
        u: list of floats
            the reduced potential at every state.
        """
        state = self.context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy()

        u = np.array([pot_energy / (self.kB * t) for t in self.temperature_ladder])

        return u

    def _sample_state(self):
        """
        Sample the state given the current configuration of the system

        Returns
        -------

        """
        # Sample from the states using Gibbs sampling
        q = np.exp(- self.reduced_potential() + self.biases)
        self.weights = q / np.sum(q)
        states = range(len(self.temperature_ladder))

        # Update the state and integrator with the new temperature
        self.state = np.random.choice(states, p=self.weights)
        self.temperature = self.temperature_ladder[self.state]
        self.integrator.setTemperature(self.temperature)

    def sample(self, nsteps=1000, niterations=500, save_freq=1):
        """
        Sample over configurations and states (temperatures) using Gibbs sampling.

        Parameters
        ----------
        nsteps: int
            the number of molecular dynamics steps per iteration to sample the configuration.
        niterations: int
            the number of iterations of Gibbs sampling, in which MD is performed and temperature sampling
        save_freq: int
            the frequency with which to save record the state and update the state histogram

        Returns
        -------
        current_state: numpy.ndarray
            binary vector where the 1 indicates the current state of the system.

        """
        for iteration in range(niterations):
            # Sample position
            self.integrator.step(nsteps)
            # Sample state (temperature)
            self._sample_state()
            # Save state
            if iteration % save_freq == 0:
                self.nmoves += 1
                self.histogram[self.state] += 1

        current_state = np.zeros(len(self.temperature_ladder))
        current_state[self.state] = 1
        return current_state

    def reset_statistics(self):
        """
        Reset the state histogram and move counter
        """
        self.histogram = np.zeros(len(self.temperature_ladder))
        self.nmoves = 0
