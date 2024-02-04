import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# Distribution classes to be used in the classification
distributions = [
    stats.norm,
    stats.expon,
    stats.uniform,
    stats.lognorm,
    stats.t,
    stats.chi2
]

def import_data(file: str) -> np.ndarray:
    """
    Import data from a csv file and return a pandas DataFrame.
    """
    data = np.loadtxt(file)
    return data

def qqplot(data: np.ndarray, distr: stats._continuous_distns) -> (np.ndarray, np.ndarray):
    """
    Create a QQ-plot of the data and the distribution.
    """
    # Fit the distribution to the data
    params = distr.fit(data)
    # Generate random numbers from the fitted distribution
    rv = distr(*params)
    # Create a QQ-plot via Filliben's estimate
    quantiles, _ = stats.probplot(data, dist=rv, plot=plt)
    return quantiles

def classify_AIC(data: np.ndarray) -> str:
    """
    Classify the data based on the best fitting distribution.
    """
    # Calculate the AIC for each distribution
    aic_values = []
    for distr in distributions:
        params = distr.fit(data)
        aic = distr.nnlf(params, data)
        aic_values.append(aic)
    # Find the distribution with the lowest AIC
    best_fit = distributions[np.argmin(aic_values)]
    return best_fit.name

def classify_distance(data: np.ndarray) -> str:
    """
    Classify the data based on the distribution with the smallest distance.
    """
    # Calculate the distance for each distribution
    distances = []
    for distr in distributions:
        params = distr.fit(data)
        rv = distr(*params)
        distance = stats.kstest(data, rv.cdf)[0]
        distances.append(distance)
    # Find the distribution with the smallest distance
    best_fit = distributions[np.argmin(distances)]
    return best_fit.name