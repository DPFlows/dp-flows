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

from sklearn.mixture import _gaussian_mixture
from sklearn import *

import flows
import utils
import shutil
import brewer2mpl

from scipy.io import loadmat

plot_bounds = {
  'lifesci': ((0.0, 4.0), (-6, 11.5)),
}

colors = iter(brewer2mpl.get_map('Set1', 'qualitative', 7).mpl_colors)

linestyles = iter([
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
  delta = 1 / (num_samples ** 1.1)

  print('Delta: {}'.format(delta))

  fig = plt.figure(figsize=(4, 3))
  ax = fig.add_subplot(111)
  dpi = 300

  threshold = 0.6
  num_bins = 50

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

    plotted_examples = False
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

  for name, dirname in [('DP-MoG (MA)', 'DP-EM-MA'), ('DP-MoG (zCDP)', 'DP-EM-zCDP')]:
      matfiledir = 'gmm/matfiles/' + dataset + '/3/' + dirname + '/'
      epsilons, likelihoods = [], []
      plotted_examples = False
      for filename in os.listdir(matfiledir):
        if os.path.isdir(matfiledir + '/' + filename) or filename == '.DS_Store':
          continue

        filename = matfiledir + filename
        max_iter = 10

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

        likelihood = np.mean(GMM.score_samples(X_test)).item()

        epsilons.append(epsilon)
        likelihoods.append(likelihood)

      likelihoods = [l for e, l in sorted(zip(epsilons, likelihoods))]
      epsilons = sorted(epsilons)
      plt.plot(epsilons, likelihoods, color=next(colors), linestyle=next(linestyles)[1], label=name, rasterized=True)

  xlim, ylim = plot_bounds[dataset]
  plt.xlabel(r'Cumulative Privacy Loss $\varepsilon$ ($\delta = 1.52 \times 10^{-5}$)')
  plt.ylabel('Average Log Likelihood (Test)')
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  ax.legend(loc='lower right', prop={'size': 8})
  plt.grid(True)
  plt.tight_layout()
  plt.savefig('analysis/figure_2.png'.format(dataset), dpi=dpi)
  plt.clf()
