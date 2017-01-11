"""
A set of tools that can be used in the bayesian adaptive estimation of free energies
and plotting of results.
"""
import numpy as np

#------Functions for the Gaussian mixture model example------#

def gen_free_energy(distribution, spread=2.0, location=0.0, size=1):
    """
    Generate example free energies from a specified distribution

    Parameters
    ----------
    distribution: str
        the distribution from which to draw samples
    size: int
        number of samples to generate
    spread: list or float
        the spread/scale parameter of the distribution, e.g. standard deviation for gaussian
    location: list of float
        the center of the distribution, e.d.

    Returns
    -------
    numpy.ndarray
    samples of free energy with dimentions equal to spread and location

    """
    if distribution.lower() == 'gaussian':
        f_true = np.random.normal(loc=location, scale=spread, size=size)
    elif distribution.lower() == 'cauchy':
        from scipy.stats import cauchy
        f_true = cauchy.rvs(loc=location, scale=spread, size=size)
    else:
        raise Exception('The distribution must be either "gaussain" or "cauchy".')
    return f_true

def gen_sigmas(sigma1, f):
    """
    Generate standard deviations for one-dimensional Gaussian distributions by the relative free energy of the
    normalizing constants.

    Parameters
    ----------
    sigma1: float
        the standard deviation from which all other standard deviations are calculated relative to
    f: numpy.ndarray or float
        the relative free energy of the other Gaussian distributions to sigma1

    Returns
    -------
    numpy.ndarray
        vector of standard deviations
    """
    return sigma1 * np.hstack((1.0, np.exp(-f)))

#---------Plotting tools-------#
def histomatic(data, nbins=30):
    """
    Wrapper to histogram data for plotting counts against midpoints.
    """
    counts, bins = np.histogram(data, nbins)
    midpoints = bins[0:len(bins)-1] + np.diff(bins)/2.0
    return midpoints, counts

# # These are the "Tableau" colors as RGB. Taken on 26th Nov 2015 from:
#http://tableaufriction.blogspot.co.uk/2012/11/finally-you-can-use-tableau-data-colors.html
# In order: blue, green, purple, orange. Hopefully a good compromise for colour-blind people.
tableau4 = [(31, 119, 180),(44, 160, 44),(148,103,189),(255, 127, 14)]
tableau4_light = [(174,199,232),(152,223,138),(197,176,213),(255,187,120)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau4)):
    r, g, b = tableau4[i]
    tableau4[i] = (r / 255., g / 255., b / 255.)
    r, g, b = tableau4_light[i]
    tableau4_light[i] = (r / 255., g / 255., b / 255.)