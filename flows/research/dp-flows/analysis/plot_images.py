import sys

sys.path.insert(0, './')
sys.path.insert(0, '../../')

from jax import random, nn
import configparser
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from jax.scipy.special import logsumexp

import flows
import utils
import shutil

import torchvision
import torch


def save_grid(X, path, nrow=10):
    # Append single additional pixel to BSDS300 data
    if 'bsds300' in path:
        X = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)

    X = torch.from_numpy(onp.asarray(X)).reshape(X.shape[0], 1, round(X.shape[1] ** 0.5), round(X.shape[1] ** 0.5))
    grid = torchvision.utils.make_grid(X, nrow=nrow, padding=1, normalize=True, pad_value=1.)
    npgrid = grid.cpu().numpy()
    plt.imsave(path, np.transpose(npgrid, (1, 2, 0)))


if __name__ == '__main__':
    flow_path  = 'out/mnist/flows/private/' if len(sys.argv) == 1 else sys.argv[1]

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

    _, _, X = utils.get_datasets(dataset)
    _, input_dim = X.shape

    shutil.copyfile(flow_path + 'flow_utils.py', 'analysis/flow_utils.py')
    from analysis import flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    params = pickle.load(open(flow_path + 'val_params.pkl', 'rb'))

    temp_key, key = random.split(key)
    real_samples = random.permutation(temp_key, X)[:40]

    temp_key, key = random.split(key)
    fake_samples = sample(temp_key, params, 40)

    samples = np.concatenate((fake_samples, real_samples), 0)
    save_grid(samples, flow_path + 'images.png', 10)
