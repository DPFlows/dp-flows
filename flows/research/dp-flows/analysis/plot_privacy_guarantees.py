import sys

sys.path.insert(0, './')
sys.path.insert(0, '../../')

from jax import random
import configparser
import jax.numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
import os
import pickle
from tqdm import tqdm

import flows
import utils
import shutil

plot_bounds = {
    'lifesci': ((0, 150000), (0.0, 4.0)),
}


if __name__ == '__main__':
    flow_path  = 'out/lifesci/flows/private/' if len(sys.argv) == 1 else sys.argv[1]

    key = random.PRNGKey(0)

    config_file = flow_path + 'experiment.ini'
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
    kfold = 5
    delta = 1e-4

    X_full = utils.get_datasets(dataset)

    kfold = model_selection.KFold(kfold, shuffle=True, random_state=0)
    kfold.get_n_splits(X_full)

    train_index, test_index = list(kfold.split(X_full))[0]
    X, X_test = X_full[train_index], X_full[test_index]
    num_samples, input_dim = X.shape
    delta = 1e-4 if dataset == 'lifesci' else 1. / X.shape[0]

    print('δ = {}'.format(delta))
    print('q = b / N = {}'.format(minibatch_size / num_samples))
    print('sig = {}'.format(noise_multiplier))

    print('b = {}'.format(minibatch_size))
    print('N = {}'.format(num_samples))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    iterations = sorted([int(d) for d in os.listdir(flow_path) if os.path.isdir(flow_path + d)])
    linestyle = iter(['--', '-', ':', '-.'])
    for composition in ['gdp', 'ma']:
        for sampling in ['poisson', 'uniform']:
            epsilons = []
            print('Composing in {}...'.format(composition))
            for iteration in iterations:
                epsilons.append(utils.get_epsilon(
	            private, composition, sampling, iteration,
	            noise_multiplier, num_samples, minibatch_size, delta,
                ))
            plt.plot(iterations, epsilons, label='{} {}'.format(composition.upper(), sampling.capitalize()), linestyle=next(linestyle))

    xlim, ylim = plot_bounds[dataset]
    plt.xlabel('Iterations')
    plt.ylabel('Cumulative Privacy Loss ε (δ = {:.3g})'.format(delta))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis/{}-privacy-guarantees.png'.format(dataset), dpi=600)
    plt.close('all')
