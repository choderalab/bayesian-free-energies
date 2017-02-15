import numpy as np

class SAMSAdaptor(object):
    """
    Implements the update scheme for self adjusted mixture sampling
    """
    def __init__(self, nstates, zetas=None, target_weights=None, mode='two-stage', beta=0.6, flat_hist=0.2):
        """
        Parameters
        ----------
        nstates: int
            The number of free energies to infer
        zeta: numpy array
            Simultaneously the estimate of the free energy and the applied simulation bias
        """

        self.nstates = nstates
        self.beta = beta
        self.flat_hist = flat_hist
        self.mode = mode

        if zetas is None:
            self.zetas = np.zeros(self.nstates)
        elif len(zetas) != self.nstates:
            raise Exception('The length of the  bias/estimate (zetas) array is not equal to nstates')

        if target_weights is None:
            self.target_weights = np.repeat(1.0 / nstates, nstates)
        else:
            if len(target_weights) != self.nstates:
                raise Exception('The length of the target weights array is not equal to nstates')
            else:
                self.target_weights = target_weights

        self.stage = 'burn-in'
        self.time = 1
        self.burn_in_length = None

    def _calc_gain(self, state):
        """

        :param stage:
        :param state:
        :return:
        """

        if self.mode == 'two-stage':
            if self.stage == 'burn-in':
                gain = np.min((self.target_weights[state], self.time ** (-self.beta)))
            else:
                factor = (self.time - self.burn_in_length + self.burn_in_length ** (-self.beta)) ** (-1)
                gain = np.min((self.target_weights[state], factor))
        else:
            gain = 1.0 / self.time

        return gain

    def update(self, state, noisy_observation, histogram=None):
        """

        :param current_state:
        :param histogram:
        :return:
        """
        if self.mode == 'two-stage':
            if self.stage == 'burn-in' and histogram is not None:
                # Calculate how far the histogram is from the target weights
                fraction = 1.0 * histogram / np.sum(histogram)
                dist = np.mean(np.absolute(fraction - self.target_weights) / self.target_weights)
                if dist <= self.flat_hist:
                    # If histogram appears suitably flat then switch to slow growth
                    self.stage = 'slow-growth'
                    self.burn_in_length = self.time
            elif self.stage == 'burn-in' and histogram is None:
                # If no histogram is supplied the update scheme switches to slow growth
                self.stage = 'slow-growth'
                self.burn_in_length = self.time

        gain = self._calc_gain(state)
        zetas_half = self.zetas + gain * noisy_observation
        self.zetas = zetas_half - zetas_half[0]

        self.time += 1

        return self.zetas

