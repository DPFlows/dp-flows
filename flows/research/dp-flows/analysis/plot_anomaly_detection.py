import sys

sys.path.insert(0, './')
sys.path.insert(0, '../../')

from jax import random
import configparser
import jax.numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

from sklearn import model_selection

import flows
import utils
import shutil


def get_gmm_params(path):
    mat = loadmat(path)
    means = mat['model']['cpd'][0][0][0][0][0].transpose()
    covariances = mat['model']['cpd'][0][0][0][0][1].transpose()
    weights = mat['model']['mixWeight'][0][0][0]
    epsilon = float(path.split('_')[3][8:])
    return means, covariances, weights


if __name__ == '__main__':
    key = random.PRNGKey(0)

    dataset = 'lifesci'
    flow_path = 'out/' + dataset + '/flows/private-kfold-10/0/'

    config_file = flow_path + 'experiment.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    flow = config['flow'].lower()
    minibatch_size = int(config['minibatch_size'])
    noise_multiplier = float(config['noise_multiplier'])
    num_blocks = int(config['num_blocks'])
    num_hidden = int(config['num_hidden'])
    normalization = str(config['num_hidden']).lower() == 'true'
    private = str(config['private']).lower() == 'true'
    sampling = config['sampling'].lower()
    kfold = int(config['kfold'])

    shutil.copyfile(flow_path + 'flow_utils.py', 'analysis/flow_utils.py')
    from analysis import flow_utils

    X_full = utils.get_datasets(dataset)

    kfold = model_selection.KFold(kfold, shuffle=True, random_state=0)
    kfold.get_n_splits(X_full)
    train_index, test_index = list(kfold.split(X_full))[0]
    X, X_test = X_full[train_index], X_full[test_index]
    num_samples, input_dim = X.shape
    delta = 1e-4 if dataset == 'lifesci' else 1. / X.shape[0]

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    iteration = 10000
    composition = 'ma'

    epsilon = utils.get_epsilon(
        private, composition, sampling, iteration,
        noise_multiplier, num_samples, minibatch_size, delta
    )

    params = pickle.load(open(flow_path + str(iteration) + '/params.pkl', 'rb'))
    flow_likelihoods = log_pdf(params, X_test)
    print(flow_likelihoods)

    gmm_path = 'out/' + dataset + '/gmm/' + str(n_components) + '/DP-EM-MA/'
    GMM.means_, GMM.covariances_, GMM.weights_ = means, covariances, weights
    GMM.precisions_cholesky_ = _gaussian_mixture._compute_precision_cholesky(GMM.covariances_, 'full')
    gmm_likelihoods = -GMM.score_samples(X).mean()
    print(gmm_likelihoods)

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xlim, ylim = plot_bounds[dataset]
    plt.xlabel('Cumulative Privacy Loss ε (δ = {:.3g})'.format(delta))
    plt.ylabel('Average Log Likelihood (Test)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis/{}-likelihoods.png'.format(dataset), dpi=600)
    plt.clf()
    """
