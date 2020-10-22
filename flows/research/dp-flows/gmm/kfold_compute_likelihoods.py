import sys

sys.path.insert(0, './')
sys.path.insert(0, '../../')

from sklearn import preprocessing, mixture
from sklearn.mixture import _gaussian_mixture
from jax import numpy as np
from jax import random
import numpy as np
import shutil
from tqdm import tqdm
from scipy.io import loadmat
import glob
import utils
import flows


def get_gmm_params(path):
    mat = loadmat(path)
    means = mat['model']['cpd'][0][0][0][0][0].transpose()
    covariances = mat['model']['cpd'][0][0][0][0][1].transpose()
    weights = mat['model']['mixWeight'][0][0][0]
    epsilon = float(path.split('_')[3][8:])
    return epsilon, mat['loglik_test'].reshape(-1), means, covariances, weights


def get_all_gmm_params(base_path):
    results = []
    for path in glob.glob(base_path + '*.mat'):
        if 'epsilon' in path:
            results.append(get_gmm_params(path))
    return results


def main():
    dataset = 'lifesci'
    n_components = 3
    model_name = 'DP-EM-ADV'
    matfiledir = 'gmm/matfiles/' + dataset + '/' + str(n_components) + '/' + model_name + '/'

    output_dir = ''
    for ext in ['out', dataset, 'gmm', str(n_components), model_name]:
        output_dir += ext + '/'
        utils.make_dir(output_dir)

    # epsilons, train_losses, val_losses, test_losses = [], [], [], []
    epsilons, test_losses = [], []
    for result in tqdm(sorted(get_all_gmm_params(matfiledir))):
        """
        epsilon, GMM.means_, GMM.covariances_, GMM.weights_ = result
        GMM.precisions_cholesky_ = _gaussian_mixture._compute_precision_cholesky(GMM.covariances_, 'full')

        train_loss = -GMM.score_samples(X).mean()
        val_loss = -GMM.score_samples(X_val).mean()
        test_loss = -GMM.score_samples(X_test).mean()
        X_syn = GMM.sample(X.shape[0])[0]

        epsilons.append(epsilon)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        """
        epsilons.append(result[0])
        test_losses.append(result[1].mean())

    utils.dump_obj(epsilons, output_dir + 'epsilons.pkl')
    # utils.dump_obj(train_losses, output_dir + 'train_losses.pkl')
    # utils.dump_obj(val_losses, output_dir + 'val_losses.pkl')
    utils.dump_obj(test_losses, output_dir + 'test_losses.pkl')


if __name__ == '__main__':
    main()
