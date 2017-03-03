import numpy as np
from bams.bayes_adaptor import MultinomialBayesEstimator, BayesAdaptor


class TestMultinomialEstimator(object):
    """
    Suit of functions to test the class that performs Bayesian estimation of free energies of biased multinomial
    samples.
    """

    def test_bias_initialization(self):
        """
        Assess whether the biases are correctly read in and appropriately set to zero if no biases are applied.
        """
        # Initialize the class with fake samples
        counts = np.array((10, 103, 243, 82))
        biases = np.array((0.0, 0.0, 0.0, 0.0))

        esimator_biased = MultinomialBayesEstimator(counts=counts, zetas=biases)
        esimator_unbiased = MultinomialBayesEstimator(counts=counts)

        assert np.all(esimator_unbiased.free_energies == esimator_biased.free_energies)

    def test_stacking_inputs(self):
        """
        The class for Multinomial bayesian estimation should be able to handle histograms from simulations with
        different applied biases, this is achieved with using matrices of histograms and biases.
        """
        counts = np.array(((10, 103, 243, 82), (30, 11, 23, 72)))
        biases = np.array(((10.0, 30.0, 43.0, 28.0), (4.0, 14.0, 36.0, 81.0)))

        esimator = MultinomialBayesEstimator(counts=counts, zetas=biases)

    def test_map_estimation(self):
        """
        Test if the maximum aposteriori estimator works
        """
        # Initialize the class with fake samples
        counts = np.array((10, 103, 243, 82))
        biases = np.array((10.0, 30.0, 43.0, 28.0))
        esimator = MultinomialBayesEstimator(counts=counts, zetas=biases, prior='gaussian', location=0, spread=100)

        map = esimator.map_estimator()

    def test_map_estimation_stacked_inputs(self):
        """
        Test if the maximum aposteriori estimator works when matrices are supplied for the counts and biases.
        """
        # Initialize the class with fake samples
        counts = np.array(((10, 103, 243, 82), (30, 11, 23, 72)))
        biases = np.array(((10.0, 30.0, 43.0, 28.0), (4.0, 14.0, 36.0, 81.0)))
        esimator = MultinomialBayesEstimator(counts=counts, zetas=biases, prior='gaussian', location=0, spread=100)

        map = esimator.map_estimator()

    def test_gaussian_posterior_sampling(self):
        """
        Test whether the posterior can successully be sampled from
        """
        counts = np.array((10, 103))
        biases = np.array((10.0, 30.0))
        esimator = MultinomialBayesEstimator(counts=counts, zetas=biases, prior='gaussian', location=0, spread=100)

        # Sample from the posterior
        esimator.sample_posterior(nwalkers=4, nmoves=10)


    def test_laplace_posterior_sampling(self):
        """
        Test whether the posterior can successully be sampled from
        """
        counts = np.array((10, 103))
        biases = np.array((10.0, 30.0))
        esimator = MultinomialBayesEstimator(counts=counts, zetas=biases, prior='laplace', location=0, spread=100)

        # Sample from the posterior
        esimator.sample_posterior(nwalkers=4, nmoves=10)

    def test_cauchy_posterior_sampling(self):
        """
        Test whether the posterior can successully be sampled from
        """
        counts = np.array((10, 103))
        biases = np.array((10.0, 30.0))
        esimator = MultinomialBayesEstimator(counts=counts, zetas=biases, prior='cauchy', location=0, spread=100)

        # Sample from the posterior
        esimator.sample_posterior(nwalkers=4, nmoves=10)


class TestBayesianAdaptor(object):
    """
    Suit of tests for the Bayesian tool that can process simulation data and output suitable biases for the next round
    of simulations.
    """
    def test_subclass(self):
        """
        Test whether BayesAdaptor is has inherinted the properties of MultinomialBayesEstimator. If so, then some of
        the tests for MultinomialBayesEstimator automatically apply to BayesAdaptor.
        """
        assert issubclass(BayesAdaptor, MultinomialBayesEstimator)

    def test_initialization(self):
        """
        See whether if the class can correctly initialize
        """
        counts = np.array((10, 103, 243, 82))
        adaptor = BayesAdaptor(counts=counts)

    def test_map_update(self):
        """
        Ensure the maximum a posteriori (MAP) update scheme is working and produces updates with the correct dimension.
        """
        counts = np.array(((10, 103), (30, 11)))
        current_biases = np.array(((10.0, 30.0), (4.0, 14.0)))
        adaptor = BayesAdaptor(counts=counts, zetas=current_biases, method='map')
        next_biases = adaptor.update(sample_posterior=False)

        assert len(next_biases) == counts.shape[0]

    def test_quantitative_map_update(self):
        """
        A rough quantitative test for the MAP estimate. When there are far more counts in one state than another, the
        biases should try to ensure the histogram is flatter in the next round.
        """
        counts = np.array((10, 1000))
        adaptor = BayesAdaptor(counts=counts, method='map')
        biases = adaptor.update(sample_posterior=False)

        assert biases[0] > biases[1]

    def test_thompson_update(self):
        """
        Ensure the Thompson update scheme is working and produces updates with the correct dimension.
        """
        counts = np.array(((10, 103), (30, 11)))
        current_biases = np.array(((10.0, 30.0), (4.0, 14.0)))
        adaptor = BayesAdaptor(counts=counts, zetas=current_biases, method='thompson')
        next_biases = adaptor.update(sample_posterior=True, nwalkers=4, nmoves=10, burnin=0)

        assert len(next_biases) == counts.shape[0]

    def test_mean_update(self):
        """
        Ensure the mean update scheme is working and produces updates with the correct dimension.
        """
        counts = np.array(((10, 103), (30, 11)))
        current_biases = np.array(((10.0, 30.0), (4.0, 14.0)))
        adaptor = BayesAdaptor(counts=counts, zetas=current_biases, method='mean')
        next_biases = adaptor.update(sample_posterior=True, nwalkers=4, nmoves=10, burnin=0)

        assert len(next_biases) == counts.shape[0]

    def test_median_update(self):
        """
        Ensure median update scheme is working and produces updates with the correct dimension.
        """
        counts = np.array(((10, 103), (30, 11)))
        current_biases = np.array(((10.0, 30.0), (4.0, 14.0)))
        adaptor = BayesAdaptor(counts=counts, zetas=current_biases, method='median')
        next_biases = adaptor.update(sample_posterior=True, nwalkers=4, nmoves=10, burnin=0)

        assert len(next_biases) == counts.shape[0]