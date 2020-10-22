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
from scipy.io import loadmat

from sklearn import *
from sklearn.mixture import _gaussian_mixture

import flows
import utils
import shutil
import brewer2mpl


if __name__ == '__main__':
  experiment_path = 'out/lifesci/flows/private-kfold-10/'
  piece = 0
  composition = 'ma'

  key = random.PRNGKey(0)

  shutil.copyfile(experiment_path + str(piece) + '/flow_utils.py', 'analysis/flow_utils.py')
  shutil.copyfile(experiment_path + str(piece) + '/experiment.ini', 'experiment.ini')

  from config import *
  from analysis import flow_utils

  dataset = 'lifesci'

  X_full = utils.get_datasets(dataset)

  kfold = model_selection.KFold(kfold, shuffle=True, random_state=0)
  kfold.get_n_splits(X_full)

  train_idx, test_idx = list(kfold.split(X_full))[piece]
  X, X_test = X_full[train_idx], X_full[test_idx]
  delta = 1. / (X.shape[0] ** 1.1)

  #model_class = linear_model.LinearRegression
  model_class = neighbors.KNeighborsRegressor
  #model_class = neural_network.MLPRegressor

  #metric = metrics.median_absolute_error
  metric = metrics.mean_squared_error
  #metric = metrics.r2_score

  for dim in range(X_test.shape[1]):
    colors = iter(brewer2mpl.get_map('Set1', 'qualitative', 7).mpl_colors)

    #real_inputs, real_outputs = X[:, :-1], X[:, -1]
    #target_inputs, target_outputs = X_test[:, :-1], X_test[:, -1]

    real_inputs, real_outputs = np.concatenate((X[:, :dim], X[:, dim+1:]), 1), X[:, dim]
    target_inputs, target_outputs = np.concatenate((X_test[:, :dim], X_test[:, dim+1:]), 1), X_test[:, dim]

    plt.close('all')

    plt.rc('font', family='sans-serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    #plt.rc('text', usetex=True)

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)

    # -- DP-MoG --

    path = 'gmm/matfiles/lifesci/3/DP-EM-MA/'
    epsilons, performances = [], []

    for filename in os.listdir(path):
      filename = path + filename # 'train_K=3_G=lap_0_epsilon=2_delta=1.5157e-05_comp=4.mat'

      if filename.split('/')[-1] == '.DS_Store' or os.path.isdir(filename):
        continue

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

      X_fake = GMM.sample(X.shape[0])[0]
      fake_inputs, fake_outputs = X_fake[:, :-1], X_fake[:, -1]

      model = model_class()
      model.fit(fake_inputs, fake_outputs)

      y_pred = model.predict(target_inputs)
      performance = metric(target_outputs, y_pred)
      density = GMM.score_samples(X_test).mean()
      print('DP-MoG: Perf: {:.3f}, Dens: {:.3f}'.format(performance, density))

      epsilons.append(epsilon)
      performances.append(performance)

    epsilons, performances = zip(*sorted(zip(epsilons, performances)))
    ax.plot(epsilons, performances, color=next(colors), linestyle=':', label='DP-MoG', rasterized=True)

    # ---------

    # -- MoG --

    n_components = 8
    GMM = mixture.GaussianMixture(n_components)
    GMM.fit(X)

    X_fake = GMM.sample(X.shape[0])[0]
    fake_inputs, fake_outputs = X_fake[:, :-1], X_fake[:, -1]

    model = model_class()
    model.fit(fake_inputs, fake_outputs)

    y_pred = model.predict(target_inputs)
    performance = metric(target_outputs, y_pred)
    density = GMM.score_samples(X_test).mean()
    print('MoG: Perf: {:.3f}, Dens: {:.3f}'.format(performance, density))

    ax.plot(epsilons, [performance for _ in range(len(epsilons))], color=next(colors), linestyle='-.', label='MoG', rasterized=True)

    # ---------

    # -- DP-NF --

    experiment_path = 'out/lifesci/flows/private-kfold-10/'
    piece = 0
    composition = 'ma'

    shutil.copyfile(experiment_path + str(piece) + '/flow_utils.py', 'analysis/flow_utils.py')
    shutil.copyfile(experiment_path + str(piece) + '/experiment.ini', 'experiment.ini')

    from config import *
    from analysis import flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    #prior = utils.get_prior(prior_type)
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, X.shape[1])

    epsilons, performances = [], []
    for iteration in sorted([int(f.name) for f in os.scandir(experiment_path + str(piece) + '/') if f.is_dir()]):
      epsilon = utils.get_epsilon(
        private, composition, sampling, iteration,
        noise_multiplier, X.shape[0], minibatch_size, delta
      )

      params_path = experiment_path + str(piece) + '/' + str(iteration) + '/params.pkl'
      params = pickle.load(open(params_path, 'rb'))

      temp_key, key = random.split(key)
      X_fake = sample(temp_key, params, X.shape[0])
      fake_inputs, fake_outputs = X_fake[:, :-1], X_fake[:, -1]

      model = model_class()
      model.fit(fake_inputs, fake_outputs)

      y_pred = model.predict(target_inputs)
      performance = metric(target_outputs, y_pred)
      density = log_pdf(params, X_test).mean()
      print('DP-NF: Perf: {:.3f}, Dens: {:.3f}'.format(performance, density))

      epsilons.append(epsilon)
      performances.append(performance)

    ax.plot(epsilons, performances, color=next(colors), linestyle='-', label='DP-NF', rasterized=True)

    # ---------


    # -- DP-NF (infinity) --

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    #prior = utils.get_prior(prior_type)
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, X.shape[1])

    experiment_path = 'out/lifesci/flows/nonprivate'
    params_path = experiment_path + '/test_params.pkl'
    params = pickle.load(open(params_path, 'rb'))

    shutil.copyfile(experiment_path + '/flow_utils.py', 'analysis/flow_utils.py')
    shutil.copyfile(experiment_path + '/experiment.ini', 'experiment.ini')

    from config import *
    from analysis import flow_utils

    temp_key, key = random.split(key)
    X_fake = sample(temp_key, params, X.shape[0])
    fake_inputs, fake_outputs = X_fake[:, :-1], X_fake[:, -1]

    model = model_class()
    model.fit(fake_inputs, fake_outputs)

    y_pred = model.predict(target_inputs)
    performance = metric(target_outputs, y_pred)
    density = log_pdf(params, X_test).mean()
    print('NF: Perf: {:.3f}, Dens: {:.3f}'.format(performance, density))

    ax.plot(epsilons, [performance for _ in range(len(epsilons))], color=next(colors), linestyle='--', label='NF', rasterized=True)

    # -----------

    # -- Baseline --

    model = model_class()
    model.fit(real_inputs, real_outputs)

    y_pred = model.predict(target_inputs)
    performance = metric(target_outputs, y_pred)
    print('Baseline: {:.3f}'.format(performance))

    ax.plot(epsilons, [performance for _ in range(len(epsilons))], color=next(colors), linestyle=(0, (3, 5, 1, 5, 1, 5)), label='Baseline', rasterized=True)

  # ------

    if dataset == 'lifesci':
      if dim == 0:
        pass
      else:
        ax.set_xlim(0.15, 2.75)
        ax.set_ylim(0.0, 0.018)

    ax.legend(loc='upper right')
    ax.set_xlabel(r'Cumulative Privacy Loss $\varepsilon$ ($\delta$  = 1.52e-05)')
    ax.set_ylabel('Downstream Regressor MSE')
    fig.tight_layout()
    fig.savefig('downstream_task/{}.png'.format(dim), dpi=400)
