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
import numpy as onp

import flows
import utils
import shutil


if __name__ == '__main__':
    plot_bounds = {
        'lifesci': ((0., 4.0), (-7.5, 12.0)),
    }

    flow_path = 'out/lifesci/flows/private-kfold-10/'

    key = random.PRNGKey(0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    config_file = flow_path + '0/experiment.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    flow = config['flow'].lower()
    dataset = config['dataset'].lower()
    minibatch_size = int(config['minibatch_size'])
    noise_multiplier = float(config['noise_multiplier'])
    normalization = str(config['normalization']).lower() == 'true'
    num_blocks = int(config['num_blocks'])
    num_hidden = int(config['num_hidden'])
    private = str(config['private']).lower() == 'true'
    sampling = config['sampling'].lower()
    kfold = int(config['kfold'])

    X_full = utils.get_datasets(dataset)

    kfold = model_selection.KFold(kfold, shuffle=True, random_state=0)
    kfold.get_n_splits(X_full)

    train_index, test_index = list(kfold.split(X_full))[0]
    X, X_test = X_full[train_index], X_full[test_index]
    num_samples, input_dim = X.shape
    delta = 1e-4 if dataset == 'lifesci' else 1. / X.shape[0]

    shutil.copyfile(flow_path + '0/flow_utils.py', 'analysis/flow_utils.py')
    from analysis import flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    likelihoods, likelihoods_std = [], []
    iterations = sorted([int(d) for d in os.listdir(flow_path + '0/') if os.path.isdir(flow_path + '0/' + d)])
    for i, composition in enumerate(['gdp', 'ma']):
        print('Composing in {}...'.format(composition))

        epsilons = []
        for iteration in tqdm(iterations):
            if i == 0:
                iter_likelihoods = []

                for fold_dir in os.listdir(flow_path):
                    if fold_dir == '.DS_Store':
                        continue
                    train_index, test_index = list(kfold.split(X_full))[int(fold_dir)]
                    X, X_test = X_full[train_index], X_full[test_index]
                    delta = 1e-4 if dataset == 'lifesci' else 1. / X.shape[0]

                    try:
                        params = pickle.load(open(flow_path + fold_dir + '/' + str(iteration) + '/params.pkl', 'rb'))
                        likelihood = log_pdf(params, X_test).mean().item()
                        iter_likelihoods.append(likelihood)
                    except: pass

                likelihoods.append(onp.mean(iter_likelihoods))
                likelihoods_std.append(onp.std(iter_likelihoods))

            epsilons.append(utils.get_epsilon(
    	        private, composition, sampling, iteration,
	        noise_multiplier, num_samples, minibatch_size, delta,
            ))

        plt.plot(epsilons, likelihoods, label='DP-NF ({})'.format(composition.upper()))

    n_components = 3
    gmm_paths = [
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-EM-MA/',   'DP-EM (MA)'),
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-EM-zCDP/', 'DP-EM (zCDP)'),
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-EM-ADV/',  'DP-EM (ADV)'),
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-EM-LIN/',  'DP-EM (LIN)'),
    ]

    for path, name in gmm_paths:
        likelihoods = np.array(pickle.load(open(path + 'test_losses.pkl', 'rb')))
        # likelihood_stds = np.array(pickle.load(open(path + 'test_losses.pkl', 'rb')))
        epsilons = np.array(pickle.load(open(path + 'epsilons.pkl', 'rb')))
        plt.plot(epsilons, likelihoods, label=name)

    xlim, ylim = plot_bounds[dataset]
    plt.xlabel('Cumulative Privacy Loss ε (δ = $10^{-4}$)')
    plt.ylabel('Average Log Likelihood (Test)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('analysis/{}-likelihoods.png'.format(dataset), dpi=600)
    plt.clf()

    # -------------------------------------------------------------------------------------------------

    plot_bounds = {
        'lifesci': ((0., 4.0), (7.5, 11.5)),
    }

    flow_path = 'out/lifesci/flows/private-kfold-10/'

    key = random.PRNGKey(0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    config_file = flow_path + '0/experiment.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    flow = config['flow'].lower()
    dataset = config['dataset'].lower()
    minibatch_size = int(config['minibatch_size'])
    noise_multiplier = float(config['noise_multiplier'])
    normalization = str(config['normalization']).lower() == 'true'
    num_blocks = int(config['num_blocks'])
    num_hidden = int(config['num_hidden'])
    private = str(config['private']).lower() == 'true'
    sampling = config['sampling'].lower()
    kfold = int(config['kfold'])

    X_full = utils.get_datasets(dataset)

    kfold = model_selection.KFold(kfold, shuffle=True, random_state=0)
    kfold.get_n_splits(X_full)

    train_index, test_index = list(kfold.split(X_full))[0]
    X, X_test = X_full[train_index], X_full[test_index]
    num_samples, input_dim = X.shape
    delta = 1e-4 if dataset == 'lifesci' else 1. / X.shape[0]

    shutil.copyfile(flow_path + '0/flow_utils.py', 'analysis/flow_utils.py')
    from analysis import flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    likelihoods, likelihoods_std = [], []
    iterations = sorted([int(d) for d in os.listdir(flow_path + '0/') if os.path.isdir(flow_path + '0/' + d)])
    for i, composition in enumerate(['gdp', 'ma']):
        print('Composing in {}...'.format(composition))

        epsilons = []
        for iteration in tqdm(iterations):
            if i == 0:
                iter_likelihoods = []

                for fold_dir in os.listdir(flow_path):
                    if fold_dir == '.DS_Store':
                        continue
                    train_index, test_index = list(kfold.split(X_full))[int(fold_dir)]
                    X, X_test = X_full[train_index], X_full[test_index]
                    delta = 1e-4 if dataset == 'lifesci' else 1. / X.shape[0]

                    try:
                        params = pickle.load(open(flow_path + fold_dir + '/' + str(iteration) + '/params.pkl', 'rb'))
                        likelihood = log_pdf(params, X_test).mean().item()
                        iter_likelihoods.append(likelihood)
                    except: pass

                likelihoods.append(onp.mean(iter_likelihoods))
                likelihoods_std.append(onp.std(iter_likelihoods))

            epsilons.append(utils.get_epsilon(
    	        private, composition, sampling, iteration,
	        noise_multiplier, num_samples, minibatch_size, delta,
            ))

        plt.plot(epsilons, likelihoods, label='DP-NF ({})'.format(composition.upper()))

    n_components = 3
    gmm_paths = [
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-EM-MA/',   'DP-EM (MA)'),
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-EM-zCDP/', 'DP-EM (zCDP)'),
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-EM-ADV/',  'DP-EM (ADV)'),
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-EM-LIN/',  'DP-EM (LIN)'),
    ]

    for path, name in gmm_paths:
        likelihoods = np.array(pickle.load(open(path + 'test_losses.pkl', 'rb')))
        # likelihood_stds = np.array(pickle.load(open(path + 'test_losses.pkl', 'rb')))
        epsilons = np.array(pickle.load(open(path + 'epsilons.pkl', 'rb')))
        plt.plot(epsilons, likelihoods, label=name)

    xlim, ylim = plot_bounds[dataset]
    plt.xlabel('Cumulative Privacy Loss ε (δ = $10^{-4}$)')
    plt.ylabel('Average Log Likelihood (Test)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('analysis/{}-likelihoods-cropped.png'.format(dataset), dpi=600)
    plt.clf()
