from sklearn import datasets as ds
import jax.numpy as jnp
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .. import maf_utils as util
from .. import utils as repo_utils


@repo_utils.constant_seed
def get_datasets():
    """
    dataset = MOONS()
    X = ds.make_moons(n_samples=30000, shuffle=True, noise=0.05)[0]
    X = preprocessing.StandardScaler().fit_transform(X)
    return jnp.array(X)
    """
    return jnp.array(pd.read_csv('datasets/moons/moons.csv', header=None).values)


class MOONS:
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        trn, val, tst = load_data()

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x)
        plt.show()


def load_data():
    x = ds.make_moons(n_samples=30000, shuffle=True, noise=0.05)[0]
    return x[:24000], x[24000:27000], x[27000:]

def projection(X):
    return X
