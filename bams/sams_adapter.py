import numpy as np

class SAMSAdaptor(object):
    """
    Implements the update scheme for self adjusted mixture sampling as described by Z. Tan in Journal of Computational
    and Graphical Statistics Vol. 26 , Iss. 1, 2017
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

