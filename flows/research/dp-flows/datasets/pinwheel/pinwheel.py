import jax.numpy as jnp
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from .. import utils


@utils.constant_seed
def get_datasets():
    """
    X = make_pinwheel_data(0.3, 0.05, 5, 6000, 0.25)[0]
    X = jnp.array(preprocessing.StandardScaler().fit_transform(X))
    return X
    """
    return jnp.array(pd.read_csv('datasets/pinwheel/pinwheel.csv', header=None).values)


def postprocess(X):
    return X


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    # code from Johnson et. al. (2016)
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    np.random.seed(1)

    features = np.random.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    feats = 10 * np.einsum('ti,tij->tj', features, rotations)

    data = np.random.permutation(np.hstack([feats, labels[:, None]]))

    return data[:, 0:2], data[:, 2].astype(np.int)


def perturb_data(x, noise_ratio=0.1, noise_mean=0, noise_stddev=10, seed=0):
    """
    Replace random points by random noise
    Args:
        x: dataset
        noise_ratio: ratio of datapoints to perturb
        loc: noise mean
        scale: noise stddev
    Returns:
        perturbed dataset
    """
    np.random.seed(seed)

    # choose datapoints to perturb
    N, D = x.shape
    N_noise = int(N * noise_ratio)
    noise_indices = np.random.permutation(np.arange(N))[:N_noise]

    # perturb data: add noise (normal distributed with mean=0 and stddev=10)
    x[noise_indices, :] = np.random.normal(loc=noise_mean, scale=noise_stddev, size=(N_noise, D))

    return x
