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


def save_grid(X, path):
    X = torch.from_numpy(onp.asarray(X)).resize(X.shape[0], 1, round(X.shape[1] ** 0.5), round(X.shape[1] ** 0.5))
    grid = torchvision.utils.make_grid(X, nrow=10, padding=1, normalize=True)#, pad_value=255)
    npgrid = grid.cpu().numpy()
    plt.imsave(path, np.transpose(npgrid, (1, 2, 0)))#, interpolation='nearest')


if __name__ == '__main__':
    key = random.PRNGKey(0)

    flow_path  = 'out/conditionalmnist/flows/private/' if len(sys.argv) == 1 else sys.argv[1]
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

    X, _, X_test = utils.get_datasets(dataset)
    _, input_dim = X.shape

    shutil.copyfile(flow_path + 'flow_utils.py', 'analysis/flow_utils.py')
    from analysis import flow_utils

    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()
    init_fun = flows.Flow(bijection, prior)
    temp_key, key = random.split(key)
    _, log_pdf, sample = init_fun(temp_key, input_dim)

    params = pickle.load(open(flow_path + 'test_params.pkl', 'rb'))

    labels, images = X_test[:, :10], X_test[:, 10:]

    for i in tqdm(range(10)):
        ohe = onp.zeros(10)
        ohe[i] = 1.
        ohe = np.array(ohe)

        class_labels = np.tile(ohe, (images.shape[0], 1))

        inputs = np.concatenate((class_labels, images), 1)
        log_pdfs = log_pdf(params, inputs)

        probs = onp.asarray(nn.softmax(log_pdfs))
        idx = onp.random.choice(onp.arange(images.shape[0]), 100, replace=False, p=probs)
        sampled = images[idx]
        save_grid(sampled, flow_path + '{}_sampled.png'.format(i))

        sort = images[log_pdfs.argsort()]

        lowest_likelihood = sort[:100]
        save_grid(lowest_likelihood, flow_path + '{}_lowest.png'.format(i))

        highest_likelihood = sort[-100:]
        save_grid(highest_likelihood, flow_path + '{}_highest.png'.format(i))
