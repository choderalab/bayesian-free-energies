import numpy as np

#TODO: this tool is still in progress

class WHAMAdaptor(object):
    """
    Implementation of adaptive Bayesian WHAM (ABWHAM) as described in Park, Ensign, and Pande Physical Review E 74,
    066703 (2006).
    """
    def __init__(self, weights, counts, prior=None):

        self.weights = weights
        self.counts = counts

        if prior is None:
            self.alpha = np.repeat(1, repeats=weights.shape[0])
        elif (type(prior)==float) or (type(prior)==int):
            self.alpha = np.repeat(prior, repeats=weights.shape[0])
        elif prior.shape != weights.shape:
            raise Exception('The prior must match the dimensions of the weights')

    def update(self):
        """
        The update scheme
        """
        ln_weights = np.log(self.weights) - np.log(self.weights)[0]

        A = np.sum(self.alpha)
        theta_mod = self.alpha/A

