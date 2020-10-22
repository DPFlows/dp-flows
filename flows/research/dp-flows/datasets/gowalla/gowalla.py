import numpy as np
import jax.numpy as jnp
from sklearn import preprocessing
import pandas as pd
import gzip
from .. import utils

scaler = preprocessing.StandardScaler()

@utils.constant_seed
def get_datasets():
    """
    X = []
    with gzip.open('datasets/gowalla/loc-gowalla_totalCheckins.txt.gz', 'rt') as f:
        for line in list(f)[:100000]:
            line = line.split('\t')
            lat, lon = float(line[2]), float(line[3])
            X.append((lat, lon))
    X = preprocessing.StandardScaler().fit_transform(X)
    return jnp.array(X)
    """
    return jnp.array(pd.read_csv('datasets/gowalla/gowalla.csv', header=None).values)


def projection(X):
    return X
