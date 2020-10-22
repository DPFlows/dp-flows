import sys

sys.path.insert(0, './')
sys.path.insert(0, '../../')

from jax import random
import configparser
import matplotlib.pyplot as plt
import os
import pickle
import shutil
from tqdm import tqdm
import numpy as onp
import seaborn as sns
from sklearn import decomposition, model_selection

import flows
import utils


if __name__ == '__main__':
    key = random.PRNGKey(0)
    path = 'out/lifesci/flows/private-kfold-10/0/'

    config_file = path + 'experiment.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    composition = 'ma'
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

    delta = 1e-4 if dataset == 'lifesci' else 1 / num_samples
    print('Delta: {}'.format(delta))
    dist_bounds = ((-0.5, 0.5), (-0.5, 0.5))
    dim_bounds = (-1.0, 1.0)
    use_seaborn = False
    plot_marginals = True
    dpi = 300

    projection = decomposition.PCA()
    X_proj = projection.fit_transform(X_test)
    
    # Plot real samples
    plt.hist2d(X_proj[:, 0], X_proj[:, 1], bins=50, range=dist_bounds)
    # plt.title('{}'.format(dataset.capitalize()))
    plt.savefig(path + 'samples.png', dpi=dpi)
    plt.close('all')

    shutil.copyfile(path + 'flow_utils.py', 'analysis/flow_utils.py')
    import analysis.flow_utils as flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)

    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    for iteration in tqdm(os.listdir(path)):
        if os.path.isdir(path + iteration):
            iteration_dir = path + iteration + '/'

            epsilon = utils.get_epsilon(
                private, composition, sampling, int(iteration),
                noise_multiplier, num_samples, minibatch_size, delta
            )

            params = pickle.load(open(iteration_dir + '/params.pkl', 'rb'))

            temp_key, key = random.split(key)
            X_syn = sample(temp_key, params, X_test.shape[0])
            X_syn_proj = projection.fit_transform(X_syn)

            # Plot samples
            plt.hist2d(X_syn_proj[:, 0], X_syn_proj[:, 1], bins=50, range=dist_bounds)
            # plt.title('{} (ε  = {:g}, δ = {:g})'.format(dataset.capitalize(), epsilon, delta))
            plt.savefig(iteration_dir + '{}.png'.format(str(epsilon).replace('.', '_')), dpi=dpi)
            plt.close('all')

            # Plot marginals
            if plot_marginals:
                for dim in range(X_syn.shape[1]):
                    if use_seaborn:
                        ax = sns.distplot(X_test[:, dim], hist=False, label='Real', kde_kws={'shade': True,'linestyle':'--'})
                        ax.set_xlim(*dim_bounds)
                        ax = sns.distplot(X_syn[:, dim], hist=False, label='Synthetic', kde_kws={'shade': True})
                        ax.set_xlim(*dim_bounds)
                    else:
                        fig = plt.figure()
                        ax = fig.add_subplot()
                        ax.hist(X_test[:, dim], color='blue', alpha=0.6, bins=100, range=dim_bounds, label='Real')
                        ax.hist(X_syn[:, dim], color='orange', alpha=0.6, bins=100, range=dim_bounds, label='Synthetic')

                    # plt.title('{} [{}] (ε  = {:g}, δ = {:g})'.format(dataset.capitalize(), dim, epsilon, delta))
                    plt.legend()
                    plt.grid(True)

                    plt.savefig(iteration_dir + str(dim) + '.png', dpi=dpi)
                    plt.close('all')
