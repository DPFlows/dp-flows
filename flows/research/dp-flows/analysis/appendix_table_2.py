import sys

sys.path.insert(0, './')
sys.path.insert(0, '../../')

from jax import random
import configparser
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from sklearn import model_selection
import os
import pickle
from tqdm import tqdm

import flows
import utils
import shutil


if __name__ == '__main__':
    dataset = 'moons'
    print(dataset.capitalize())
    flow_path = 'out/' + dataset + '/flows/private-kfold-10/'
    eps_limits = [0.75, 1.50, 2.25, 3.00]

    key = random.PRNGKey(0)

    config_file = flow_path + '0/experiment.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    composition = config['composition'].lower()
    dataset = config['dataset'].lower()
    flow = config['flow'].lower()
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
    print('Delta: {}'.format(delta))

    shutil.copyfile(flow_path + '0/flow_utils.py', 'analysis/flow_utils.py')
    from analysis import flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    iterations = sorted([int(d) for d in os.listdir(flow_path + '0/') if os.path.isdir(flow_path + '0/' + d)])
    folds = list(map(int, filter(lambda d: d != '.DS_Store', os.listdir(flow_path))))

    for composition in ['gdp', 'ma']:
        print('DP-NF ({})'.format(composition.upper()))

        # Model selection not done in differentially private manner
        best_likelihoods = [None for _ in folds]
        for iteration_index, iteration in enumerate(iterations):
            likelihoods = []
            for fold in folds:
                train_index, test_index = list(kfold.split(X_full))[fold]
                X, X_test = X_full[train_index], X_full[test_index]

                try:
                    params = pickle.load(open(flow_path + str(fold) + '/' + str(iteration) + '/params.pkl', 'rb'))
                    likelihood = log_pdf(params, X_test).mean().item()
                    if not best_likelihoods[fold] or likelihood > best_likelihoods[fold]:
                        best_likelihoods[fold] = likelihood
                except: pass

                likelihoods.append(best_likelihoods[fold])

            if iteration_index < len(iterations) - 1:
                epsilon = utils.get_epsilon(
    	            private, composition, sampling, iteration,
	            noise_multiplier, num_samples, minibatch_size, delta,
                )

                next_epsilon = utils.get_epsilon(
    	            private, composition, sampling, iterations[iteration_index + 1],
	            noise_multiplier, num_samples, minibatch_size, delta,
                )

                for eps_limit in eps_limits:
                    if epsilon <= eps_limit and eps_limit < next_epsilon:
                        print('ε: {:.2g}\t${:.2f} \pm {:.2f}$'.format(epsilon, onp.mean(likelihoods), onp.std(likelihoods)))

    # -----------------------------------------------------------------------------------------------------

    dataset = 'gaussian'
    print(dataset.capitalize())
    flow_path = 'out/' + dataset + '/flows/private-kfold-10/'
    eps_limits = [0.75, 1.50, 2.25, 3.00]

    key = random.PRNGKey(0)

    config_file = flow_path + '0/experiment.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    composition = config['composition'].lower()
    dataset = config['dataset'].lower()
    flow = config['flow'].lower()
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
    print('Delta: {}'.format(delta))

    shutil.copyfile(flow_path + '0/flow_utils.py', 'analysis/flow_utils.py')
    from analysis import flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    iterations = sorted([int(d) for d in os.listdir(flow_path + '0/') if os.path.isdir(flow_path + '0/' + d)])
    folds = list(map(int, filter(lambda d: d != '.DS_Store', os.listdir(flow_path))))

    for composition in ['gdp', 'ma']:
        print('DP-NF ({})'.format(composition.upper()))

        # Model selection not done in differentially private manner
        best_likelihoods = [None for _ in folds]
        for iteration_index, iteration in enumerate(iterations):
            likelihoods = []
            for fold in folds:
                train_index, test_index = list(kfold.split(X_full))[fold]
                X, X_test = X_full[train_index], X_full[test_index]

                try:
                    params = pickle.load(open(flow_path + str(fold) + '/' + str(iteration) + '/params.pkl', 'rb'))
                    likelihood = log_pdf(params, X_test).mean().item()
                    if not best_likelihoods[fold] or likelihood > best_likelihoods[fold]:
                        best_likelihoods[fold] = likelihood
                except: pass

                likelihoods.append(best_likelihoods[fold])

            if iteration_index < len(iterations) - 1:
                epsilon = utils.get_epsilon(
    	            private, composition, sampling, iteration,
	            noise_multiplier, num_samples, minibatch_size, delta,
                )

                next_epsilon = utils.get_epsilon(
    	            private, composition, sampling, iterations[iteration_index + 1],
	            noise_multiplier, num_samples, minibatch_size, delta,
                )

                for eps_limit in eps_limits:
                    if epsilon <= eps_limit and eps_limit < next_epsilon:
                        print('ε: {:.2g}\t${:.2f} \pm {:.2f}$'.format(epsilon, onp.mean(likelihoods), onp.std(likelihoods)))

