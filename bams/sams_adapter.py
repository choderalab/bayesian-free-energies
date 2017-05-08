import numpy as np

class SAMSAdaptor(object):
    """
    Implements the update scheme for self adjusted mixture sampling as described by Z. Tan in [1]. To function, this
    class must be paired with a method to perform mixture sampling over states and configurations.

    [1] Journal of Computational and Graphical Statistics Vol. 26 , Iss. 1, 2017

    Example
    -------
    Calculating relative free energies of different Gaussian distributions using the Rao-Blackwellized scheme. To
    simplify things, this example will use the examples_systems.GaussianMixtureSampler to sample over the different
    Gaussian distributions. We know the analytical free energies for this case, but it serves to demonstrate the
    basic functionality this class.

    Initialize the distributions you can to calculate the free energies between. All free energies will be calculated
    relative to the first state (index=0). For the Gaussian example, we'll specify the standard deviations of the
    distributions, which all have a mean of zero.

    >>> sigms = np.array((1.0, 10.0, 100.))
    >>> sampler = GaussianMixtureSampler(sigmas=sigmas)

    GaussianMixtureSampler can sample over configurations and states (distributions with different standard deviations
    in this case). To use this class for your own cases, you'll have to write your own mixture sampler.

    Next, we'll initialize this class, which will estimates given samples from states.

    >>> adaptor = SAMSAdaptor(nstates=len(sigmas), two_stage=True, flat_hist=0.2)

    With the two_stage flag set to True, a burn-in scheme is performed as described in equation 15 of [1]. The burn-in
    stage will finish when the state count histogram is within 20% (flat_hist=0.2) of the target_weights. By default,
    the target_weights are uniform over the states.

    The SAMSAdaptor works by tracking the state of the sampler and updating the bias accordingly. The bias is used to
    sample from the different states at the target probability. There are 4 main stages to calculating free energies
    with this class. The example below performs 50 iterations of mixture sampling and SAMS updates and each stage is
    labeled and described below.

    >>> for i in range(500):
    >>>     sampler.step()                 # 1.
    >>>     noisy = sampler.weights        # 2.
    >>>     z = -adaptor.update(state=sampler.state , noisy_observation=noisy, histogram=sampler.histogram)     # 3.
    >>>     sampler.zetas = z              # 4.

    In stage 1., states and configurations are sampled over with GaussianMixtureSampler. In stage 2., the Rao-
    Blackwellized weights (which are proportional to the conditional probability of being sampled from one the
    Gaussians given the configuration) are calculated. If instead of the Rao-Blackwellized scheme, the SAMS binary
    scheme was being used, the 'noisy' variable would be a binary vector with the only 1 at the index of the current
    state. In stage 3., the current state of the mixture sampler ('sampler.state', which is either 0, 1, 2 in this
    case), the noisy observation, and histogram of the state counts is supplied to the SAMS object. The state counts are
    not essential here, but are necessary if you want to use the two-stage scheme. The new bias required to achieve the
    target weights for that iteration is returned. In stage 4., the bias is supplied to the sampler. Thus, the next
    iteration samples from the mixture with an updated state bias.

    As the number of iterations tends to infinity, the bias will converge to a value that can be used as an unbiased
    estimate of the free energies. When the target weights are equal, as they are above, the bias tends to the relative
    free energies of the Gaussian probability distributions.

    """
    def __init__(self, nstates, zetas=None, target_weights=None, two_stage=True, beta=0.6, flat_hist=0.2):
        """
        Parameters
        ----------
        nstates: int
            The number of free energies to infer
        zeta: numpy array
            The estimate of the free energy and the current state biasing potential
        target_weights: numpy array
            vector of the state probabilities that the sampler should converge to.
        two_stage: bool
            whether to perform the two-stage update procedure as outline by Z. Tan in Journal of Computational and
            Graphical Statistics Vol. 26 , Iss. 1, 2017. If true, the zeta parameters are adapted faster
        beta: float
            exponent of the gain during the bunr-in phase of the two-stage procedure. Should be between 0.5 and 1.0
        flat_hist: float
            degree of deviation that the state histogram can be from the target weights before the burn-in period
            in the two stage procedure ends. It is the maximum relative difference a histogram element can be
            from the respective target weight.
        """

        self.nstates = nstates
        self.beta = beta
        self.flat_hist = flat_hist
        self.two_stage = two_stage
        self.burnin = True
        self.time = 0
        self.burnin_length = None

        if zetas is None:
            self.zetas = np.zeros(self.nstates)
        elif len(zetas) != self.nstates:
            raise Exception('The length of the  bias/estimate (zetas) array is not equal to nstates')
        else:
            self.zetas = zetas

        if target_weights is None:
            self.target_weights = np.repeat(1.0 / nstates, nstates)
        else:
            if len(target_weights) != self.nstates:
                raise Exception('The length of the target weights array is not equal to nstates')
            elif np.abs(np.sum(target_weights) - 1.0) > 0.000001:
                raise Exception('The target weights do not sum to 1.')
            else:
                self.target_weights = target_weights

    def _calc_gain(self, state):
        """
        Calculates the gain factor for update.

        Parameter
        ---------
        state: int
            the index corresponding to the current state of the sampler

        Returns
        -------
        gain:
            the factor applied to the SAMS derived noisy variable
        """

        if self.two_stage:
            if self.burnin:
                gain = np.min((self.target_weights[state], self.time ** (-self.beta)))
            else:
                factor = (self.time - self.burnin_length + self.burnin_length ** (-self.beta)) ** (-1)
                gain = np.min((self.target_weights[state], factor))
        else:
            gain = 1.0 / self.time

        return gain

    def update(self, state, noisy_observation, histogram=None):
        """
        Update the estimate of the free energy based on the current state of the sample using either the binary or
        Rao-Blackwellized schemes, both of these schemes differ only by their noisy observable.

        Parameters
        ----------
        state: int
            the index corresponding to the current state of the sampler
        noisy_observation: numpy array
            the vector that will be multiplied by the gain factor when updating zeta
        histogram:
            the counts in each state collected over the simulation. Used to decide when to switch to the slow-growth
            stage if self.two_stage=True. If None, then slow-growth is automatically assumed.

        Returns
        -------
        zetas: numpy array
            the updated estimates for the free energies

        """
        # Ensure the internal clock is updated. Used for calculating the gain factor.
        self.time += 1

        if self.two_stage:
            if self.burnin and histogram is not None:
                # Calculate how far the histogram is from the target weights
                fraction = 1.0 * histogram / np.sum(histogram)
                dist = np.max(np.absolute(fraction - self.target_weights) / self.target_weights)
                if dist <= self.flat_hist:
                    # If histogram appears suitably flat then switch to slow growth
                    self.burnin = False
                    self.burnin_length = self.time
            elif self.burnin and histogram is None:
                # If no histogram is supplied the update scheme switches to slow growth
                self.burnin = False
                self.burnin_length = self.time

        gain = self._calc_gain(state)
        zetas_half = self.zetas + gain * (noisy_observation / self.target_weights)
        self.zetas = zetas_half - zetas_half[0]

        return self.zetas

