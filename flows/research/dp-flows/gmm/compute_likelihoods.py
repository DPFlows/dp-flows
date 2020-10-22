import sys

sys.path.insert(0, './')
sys.path.insert(0, '../../')

import configparser
from sklearn import preprocessing, mixture
from sklearn.mixture import _gaussian_mixture
from jax import numpy as np
from jax import random
import numpy as np
import shutil
from tqdm import tqdm

import utils
import dpem
import flows


def main(config):
    dataset = config['dataset'].lower()
    max_iter = int(config['max_iter'])
    n_components = int(config['n_components'])
    log = str(config['log']).lower() == 'true'
    dirname = config['dirname']
    matfiledir = config['matfiledir']

    X, X_val, X_test = utils.get_datasets(dataset)

    GMM = mixture.GaussianMixture(n_components=n_components, max_iter=max_iter)

    output_dir = ''
    for ext in ['out', dataset, 'gmm', dirname]:
        output_dir += ext + '/'
        utils.make_dir(output_dir)

    shutil.copyfile('experiment.ini', output_dir + 'experiment.ini')

    epsilons, train_losses, val_losses, test_losses = [], [], [], []
    for result in tqdm(sorted(dpem.get_all_gmm_params(matfiledir))):
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

        # utils.plot_dist(pca.transform(X_syn), output_dir + str(epsilon).replace('.', '_') + '.png')

    utils.dump_obj(epsilons, output_dir + 'epsilons.pkl')
    utils.dump_obj(train_losses, output_dir + 'train_losses.pkl')
    utils.dump_obj(val_losses, output_dir + 'val_losses.pkl')
    utils.dump_obj(test_losses, output_dir + 'test_losses.pkl')

    # utils.plot_dist(pca.transform(X_val), output_dir + 'real.png')

    """
    if X.shape[1] == 2:
        utils.plot_dist(X, output_dir + 'real.png')
        utils.plot_dist(X_syn, output_dir + 'synthetic.png')

    utils.plot_marginals(X_syn, output_dir, overlay=X)
    """

    return {'nll': (val_loss, 0.)}


if __name__ == '__main__':
    config_file = 'gmm/experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    print('Validation loss: {}'.format(main(config)))
