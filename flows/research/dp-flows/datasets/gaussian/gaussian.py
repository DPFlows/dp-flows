import numpy as np
import jax.numpy as jnp
import pandas as pd
from sklearn import preprocessing
from jax import random
from .. import utils


@utils.constant_seed
def get_datasets():
    """
    samples_per_component = 3750
    X = np.r_[
        np.random.randn(samples_per_component, 2) * 0.2 + np.array([3, 1]),
        np.random.randn(samples_per_component, 2) * 0.2 + np.array([3 - 1.414, 3 - 1.414]),
        np.random.randn(samples_per_component, 2) * 0.2 + np.array([1, 3]),
        np.random.randn(samples_per_component, 2) * 0.2 + np.array([3 - 1.414, 3 + 1.414]),
        np.random.randn(samples_per_component, 2) * 0.2 + np.array([3, 5]),
        np.random.randn(samples_per_component, 2) * 0.2 + np.array([3 + 1.414, 3 + 1.414]),
        np.random.randn(samples_per_component, 2) * 0.2 + np.array([5, 3]),
        np.random.randn(samples_per_component, 2) * 0.2 + np.array([3 + 1.414, 3 - 1.414]),
    ].astype(np.float32)
    X = preprocessing.StandardScaler().fit_transform(X)
    X = random.permutation(random.PRNGKey(0), jnp.array(X))
    return X
    """
    return jnp.array(pd.read_csv('datasets/gaussian/gaussian.csv', header=None).values)


def postprocess(X):
    return X
