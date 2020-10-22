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
from sklearn import *

import flows
import utils
import shutil


if __name__ == '__main__':
    #flow_path  = 'out/lifesci/flows/private/' if len(sys.argv) == 1 else sys.argv[1]
    flow_path = 'out/lifesci/flows/private-kfold-10/0/'

    key = random.PRNGKey(0)

    config_file = flow_path + 'experiment.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

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
    delta = 1 / (num_samples ** 1.1) # 1e-4 # 1 / num_samples
    print('Delta: {}'.format(delta))

    iterations = sorted([int(d) for d in os.listdir(flow_path) if os.path.isdir(flow_path + d)])

    print('δ = {}'.format(delta))
    for composition in ['gdp', 'ma']:
        print('Composing in {}...'.format(composition))
        for iteration in iterations:
            epsilon = utils.get_epsilon(
	        private, composition, sampling, iteration,
	        noise_multiplier, num_samples, minibatch_size, delta,
            )
            params = pickle.load(open(flow_path + str(iteration) + '/params.pkl', 'rb'))
            #likelihood = log_pdf(params, X_test).mean()
            print('ε: {:6g} iters: {}'.format(epsilon, iteration)) # \tLL: {:6g}'.format(epsilon, likelihood))
        print('-' * 30)
