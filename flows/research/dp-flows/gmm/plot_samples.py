import sys

sys.path.insert(0, './')
sys.path.insert(0, '../../')

from jax import numpy as np
from jax import random
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.mixture import _gaussian_mixture
from tqdm import tqdm
import numpy as np
import shutil
import seaborn as sns
from scipy.io import loadmat

import utils
import os
import flows


if __name__ == '__main__':
    plt.rc('font', family='sans-serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    dataset = 'lifesci'
    max_iter = 10
    n_components = 3
    dist_bounds = ((-0.5, 0.5), (-0.5, 0.5))
    dim_bounds = (-0.5, 0.5)
    plot_marginals = True
    use_seaborn = False
    dpi = 300
    kfold = 10

    X_full = utils.get_datasets(dataset)

    kfold = model_selection.KFold(kfold, shuffle=True, random_state=0)
    kfold.get_n_splits(X_full)

    train_index, test_index = list(kfold.split(X_full))[0]
    X, X_test = X_full[train_index], X_full[test_index]
    num_samples, input_dim = X.shape
    delta = 1 / (num_samples ** 1.1) # 1e-4 # 1 / num_samples

    """
    matfiledir = 'gmm/matfiles/' + dataset + '/' + dirname + '/'
    output_dir = ''
    for ext in ['out', dataset, 'gmm'] + dirname.split('/'):
        output_dir += ext + '/'
        utils.make_dir(output_dir)
    """

    """
    # Plot samples
    X_test_proj = projection(X_test)
    plt.hist2d(X_test_proj[:, 0], X_test_proj[:, 1], bins=100, range=dist_bounds)
    # plt.title(dataset.capitalize())
    plt.savefig(output_dir + 'real.png')
    plt.close('all')
    """

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

        #X_syn_proj = projection(X_syn)

        # Plot samples
        #plt.hist2d(X_syn_proj[:, 0], X_syn_proj[:, 1], bins=50, range=dist_bounds)
        # plt.title('{} (ε  = {:g}, δ = {:g})'.format(dataset.capitalize(), epsilon, delta))
        #plt.savefig(iteration_dir + '{}.png'.format(str(epsilon).replace('.', '_')), dpi=dpi)
        #plt.close('all')

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
    """

    for name, dirname in [('DP-MoG (MA)', 'DP-EM-MA')]:
        matfiledir = 'gmm/matfiles/' + dataset + '/3/' + dirname + '/'

        epsilons, likelihoods = [], []
        for filename in list(os.listdir(matfiledir)):
            if os.path.isdir(matfiledir + '/' + filename) or filename == '.DS_Store':
                continue

            print('Processing {}...'.format(filename))

            filename = matfiledir + filename # 'train_K=3_G=lap_0_epsilon=2_delta=1.5157e-05_comp=4.mat'
            max_iter = 10 # Always true, doesn't change anything

            matfile = filename.split('/')[-1]
            tokens = matfile.split('_')

            n_components = int(tokens[1].split('=')[-1])
            epsilon = float(tokens[4].split('=')[-1])
            delta = float(tokens[5].split('=')[-1])

            epsilon_as_str = str(epsilon).replace('.', '_')
            utils.make_dir(matfiledir + epsilon_as_str)

            obj = loadmat(filename)
            means = obj['model'][0][0][-1][0][0][0].T
            covariances = obj['model'][0][0][-1][0][0][1].T
            weights = obj['model'][0][0][-2][0]

            GMM = mixture.GaussianMixture(n_components=n_components, max_iter=max_iter)
            GMM.means_, GMM.covariances_, GMM.weights_ = means, covariances, weights
            GMM.precisions_cholesky_ = _gaussian_mixture._compute_precision_cholesky(GMM.covariances_, 'full')

            X_syn = GMM.sample(n_samples=X_test.shape[0])[0]

            for dim in range(X_syn.shape[1]):
                fig = plt.figure(figsize=(4, 3))

                if use_seaborn:
                    ax = sns.distplot(X_test[:, dim], hist=False, label='Real', kde_kws={'shade': True, 'linestyle':'--'})
                    ax.set_xlim(*dim_bounds)
                    ax = sns.distplot(X_syn[:, dim], hist=False, label='Synthetic', kde_kws={'shade': True, 'gridsize': 500})
                    ax.set_xlim(*dim_bounds)
                else:
                    ax = fig.add_subplot()
                    ax.hist(X_test[:, dim], color='blue', alpha=0.6, bins=100, range=dim_bounds, label='Real')
                    ax.hist(X_syn[:, dim], color='orange', alpha=0.6, bins=100, range=dim_bounds, label='Synthetic')

                # plt.title('{} [{}] (ε  = {:g}, δ = {:g})'.format(dataset.capitalize(), dim, epsilon, delta))
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                plt.savefig(matfiledir + '/' + epsilon_as_str + '/' +str(dim) + '.png', dpi=dpi)
                plt.close('all')
