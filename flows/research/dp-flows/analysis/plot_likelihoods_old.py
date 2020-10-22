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
import brewer2mpl


plot_bounds = {
    'lifesci': ((0.0, 4.0), (-7.6, 11.5)),
    #'lifesci': ((0.15, 6.0), (7.5, 12.0)),
}

colors = iter(brewer2mpl.get_map('Set1', 'qualitative', 7).mpl_colors)

linestyles = iter([
  #('solid', 'solid'),
  ('densely dotted',        (0, (1, 1))),
  ('densely dashed',        (0, (5, 1))),
  ('densely dashdotted',    (0, (3, 1, 1, 1))),
  ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),

  ('dotted',                (0, (1, 1))),
  ('dashed',                (0, (5, 5))),
  ('dashdotted',            (0, (3, 5, 1, 5))),
  ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),

  ('loosely dotted',        (0, (1, 10))),
  ('loosely dashed',        (0, (5, 10))),
  ('loosely dashdotted',    (0, (3, 10, 1, 10))),
  ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
])


if __name__ == '__main__':
    dataset = 'lifesci'
    flow_path = 'out/lifesci/flows/private-kfold-10/0/'

    key = random.PRNGKey(0)

    config_file = flow_path + 'experiment.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']

    flow = str(config['flow']).lower()
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

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    dpi = 300

    shutil.copyfile(flow_path + 'flow_utils.py', 'analysis/flow_utils.py')
    from analysis import flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    likelihoods = []
    for i, composition in enumerate(['gdp', 'ma']):
        epsilons = []

        iterations = sorted([int(d) for d in os.listdir(flow_path) if os.path.isdir(flow_path + d)])

        for iteration in tqdm(iterations):
            epsilon = utils.get_epsilon(
                private, composition, sampling, iteration,
                noise_multiplier, num_samples, minibatch_size, delta
            )
            epsilons.append(epsilon)

            if i == 0:
                params = pickle.load(open(flow_path + str(iteration) + '/params.pkl', 'rb'))
                likelihood = log_pdf(params, X_test).mean()
                likelihoods.append(likelihood)

        plt.plot(epsilons, likelihoods, color=next(colors), linestyle=next(linestyles)[1], label='DP-NF ({})'.format(composition.upper()), rasterized=True)

        """
        print('DP-NF ({})'.format(composition.upper()))
        for epsilon, likelihood in zip(epsilons, likelihoods):
          print(epsilon)
          print(likelihood)
          print()
        """

    n_components = 3
    gmm_paths = [
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-MoG-MA/',   'DP-MoG (MA)'),
        ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-MoG-zCDP/', 'DP-MoG (zCDP)'),
    #    ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-MoG-ADV/',  'DP-MoG (ADV)'),
    #    ('out/' + dataset + '/gmm/' + str(n_components) + '/DP-MoG-LIN/',  'DP-MoG (LIN)'),
    ]

    for path, name in gmm_paths:
        print(path)
        likelihoods = -np.array(pickle.load(open(path + 'test_losses.pkl', 'rb')))
        epsilons = np.array(pickle.load(open(path + 'epsilons.pkl', 'rb')))
        plt.plot(epsilons, likelihoods, color=next(colors), linestyle=next(linestyles)[1], label=name, rasterized=True)

        """
        print(name)
        for epsilon, likelihood in zip(epsilons, likelihoods):
          print(epsilon)
          print(likelihood)
          print()
        """

    xlim, ylim = plot_bounds[dataset]
    plt.xlabel(r'Cumulative Privacy Loss $\varepsilon$ ($\delta = 1.52 \times 10^{-5}$)')
    plt.ylabel('Average Log Likelihood (Test)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc='lower right', prop={'size': 8})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('analysis/{}-likelihoods.png'.format(dataset), dpi=dpi)
    plt.clf()
    train_index, test_index = list(kfold.split(X_full))[0]
    X, X_test = X_full[train_index], X_full[test_index]
    num_samples, input_dim = X.shape
    delta = 1 / (num_samples ** 1.1) # 1e-4 # 1 / num_samples

    matfiledir = 'gmm/matfiles/' + dataset + '/' + dirname + '/'
    output_dir = ''
    for ext in ['out', dataset, 'gmm'] + dirname.split('/'):
        output_dir += ext + '/'
        utils.make_dir(output_dir)

    """
    # Plot samples
    X_test_proj = projection(X_test)
    plt.hist2d(X_test_proj[:, 0], X_test_proj[:, 1], bins=100, range=dist_bounds)
    # plt.title(dataset.capitalize())
    plt.savefig(output_dir + 'real.png')
    plt.close('all')
    """

    for filename in os.listdir(matfiledir):
        filename = matfiledir + filename # 'train_K=3_G=lap_0_epsilon=2_delta=1.5157e-05_comp=4.mat'
        max_iter = 10 # Always true, doesn't change anything

        matfile = filename.split('/')[-1]
        tokens = matfile.split('_')

        n_components = int(tokens[1].split('=')[-1])
        epsilon = float(tokens[4].split('=')[-1])
        delta = float(tokens[5].split('=')[-1])

        obj = loadmat(filename)
        means = obj['model'][0][0][-1][0][0][0].T
        covariances = obj['model'][0][0][-1][0][0][1].T
        weights = obj['model'][0][0][-2][0]

        GMM = mixture.GaussianMixture(n_components=n_components, max_iter=max_iter)
        GMM.means_, GMM.covariances_, GMM.weights_ = means, covariances, weights
        GMM.precisions_cholesky_ = _gaussian_mixture._compute_precision_cholesky(GMM.covariances_, 'full')

        iteration_dir = output_dir + str(epsilon).replace('.', '_') + '/'
        utils.make_dir(iteration_dir)

        X_syn = GMM.sample(n_samples=X_test.shape[0])[0]
        """
        X_syn_proj = projection(X_syn)

        # Plot samples
        plt.hist2d(X_syn_proj[:, 0], X_syn_proj[:, 1], bins=50, range=dist_bounds)
        # plt.title('{} (ε  = {:g}, δ = {:g})'.format(dataset.capitalize(), epsilon, delta))
        plt.savefig(iteration_dir + '{}.png'.format(str(epsilon).replace('.', '_')), dpi=dpi)
        plt.close('all')
        """

        # Plot marginals
        if plot_marginals:
            for dim in range(X_syn.shape[1]):
                fig = plt.figure(figsize=(4, 3))

                if use_seaborn:
                    ax = sns.distplot(X_test[:, dim], hist=False, label='Real', kde_kws={'shade': True, 'linestyle':'--'})
                    ax.set_xlim(*dim_bounds)
                    ax = sns.distplot(X_syn[:, dim], hist=False, label='Synthetic', kde_kws={'shade': True, 'gridsize': 500})
                    ax.set_xlim(*dim_bounds)
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.hist(X_test[:, dim], color='blue', alpha=0.6, bins=100, range=dim_bounds, label='Real')
                    ax.hist(X_syn[:, dim], color='orange', alpha=0.6, bins=100, range=dim_bounds, label='Synthetic')

                # plt.title('{} [{}] (ε  = {:g}, δ = {:g})'.format(dataset.capitalize(), dim, epsilon, delta))
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(iteration_dir + str(dim) + '.png', dpi=dpi)
                plt.close('all')
