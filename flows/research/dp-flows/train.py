import os
import sys

sys.path.insert(0, '../../')

from datetime import datetime
from jax import jit, grad, partial, random, tree_util, vmap, lax, ops
from jax import numpy as np
from sklearn import model_selection, mixture
from tqdm import tqdm
import configparser
import numpy as onp
import pickle
import shutil

import dp
import flow_utils
import flows
import utils


def main(config):
    print(dict(config))
    key = random.PRNGKey(0)

    b1 = float(config['b1'])
    b2 = float(config['b2'])
    composition = config['composition'].lower()
    dataset = config['dataset']
    experiment = config['experiment']
    flow = config['flow']
    iterations = int(config['iterations'])
    kfold = int(config['kfold'])
    l2_norm_clip = float(config['l2_norm_clip'])
    log_params = None if config['log_params'].lower() == 'false' else int(config['log_params'])
    log_performance = int(config['log_performance'])
    lr = float(config['lr'])
    lr_schedule = config['lr_schedule'].lower()
    minibatch_size = int(config['minibatch_size'])
    noise_multiplier = float(config['noise_multiplier'])
    normalization = str(config['normalization']).lower() == 'true'
    num_blocks = int(config['num_blocks'])
    num_hidden = int(config['num_hidden'])
    optimizer = config['optimizer'].lower()
    private = str(config['private']).lower() == 'true'
    sampling = config['sampling'].lower()
    weight_decay = float(config['weight_decay'])
    overwrite = config['overwrite'].lower() == 'true'

    if dataset == 'conditionalmnist':
        raise Exception('Please use conditional_train.py')

    # Create dataset
    X_full = utils.get_datasets(dataset)

    kfold = model_selection.KFold(kfold, shuffle=True, random_state=0)
    kfold.get_n_splits(X_full)

    train_index, test_index = list(kfold.split(X_full))[0]
    X, X_test = X_full[train_index], X_full[test_index]
    num_samples, input_dim = X.shape
    delta = 1 / (num_samples ** 1.1) # 1e-4 # 1 / num_samples

    print('X: {}'.format(X.shape))
    print('X test: {}'.format(X_test.shape))
    print('Delta: {}'.format(delta))

    # Create flow
    modules = flow_utils.get_modules(flow, num_blocks, normalization, num_hidden)
    bijection = flows.Serial(*tuple(modules))
    prior = flows.Normal()

    output_dir_tokens = ['out', dataset, 'flows', experiment]

    # Start where last left off
    if log_params and overwrite:
        try:
            shutil.rmtree('/'.join(output_dir_tokens))
        except: pass

    #delta = 1e-4 if dataset == 'lifesci' else 1. / X.shape[0]

    """
    gmm = mixture.GaussianMixture(n_components=10)
    gmm.fit(X)
    prior = flows.GMM(gmm.means_, gmm.covariances_, gmm.weights_)
    """

    init_fun = flows.Flow(lambda key, shape: bijection(key, shape), prior)
    temp_key, key = random.split(key)
    params, log_pdf, sample = init_fun(temp_key, X.shape[1])

    def l2(pytree):
        leaves, _ = tree_util.tree_flatten(pytree)
        return np.sqrt(sum(np.vdot(x, x) for x in leaves))

    def loss(params, inputs):
        return -log_pdf(params, inputs).mean()

    def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, minibatch_size):
        def _clipped_grad(params, single_example_batch):
            single_example_batch = np.expand_dims(single_example_batch, 0)
            grads = grad(loss)(params, single_example_batch)
            nonempty_grads, tree_def = tree_util.tree_flatten(grads)
            total_grad_norm = np.linalg.norm([np.linalg.norm(neg.ravel()) for neg in nonempty_grads])
            divisor = lax.stop_gradient(np.amax((total_grad_norm / l2_norm_clip, 1.)))
            normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
            return tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)

        px_clipped_grad_fn = vmap(partial(_clipped_grad, params))
        std_dev = l2_norm_clip * noise_multiplier
        noise_ = lambda n: n + std_dev * random.normal(rng, n.shape)
        normalize_ = lambda n: n / float(minibatch_size)
        sum_ = lambda n: np.sum(n, 0)
        aggregated_clipped_grads = tree_util.tree_map(sum_, px_clipped_grad_fn(batch))
        noised_aggregated_clipped_grads = tree_util.tree_map(noise_, aggregated_clipped_grads)
        normalized_noised_aggregated_clipped_grads = tree_util.tree_map(normalize_, noised_aggregated_clipped_grads)
        return normalized_noised_aggregated_clipped_grads

    @jit
    def private_update(rng, i, opt_state, batch):
        params = get_params(opt_state)
        grads = private_grad(params, batch, rng, l2_norm_clip, noise_multiplier, minibatch_size)
        return opt_update(i, grads, opt_state)

    @jit
    def update(rng, i, opt_state, batch):
        params = get_params(opt_state)
        grads = grad(loss)(params, batch)
        return opt_update(i, grads, opt_state)

    pbar_range = range(1, iterations + 1)

    # Create directories and load previous params if applicable
    if log_params:
        # Create experiment directory
        output_dir = ''
        for ext in output_dir_tokens:
            output_dir += ext + '/'
            utils.make_dir(output_dir)

        # Load last params if any exist
        param_dirs = sorted([int(path) for path in os.listdir(output_dir) if os.path.isdir(output_dir + path)])
        if len(param_dirs) > 0:
            last_iteration = param_dirs[-1]
            print('Loading params from {}...'.format(str(last_iteration) + '/params.pkl'))
            params = pickle.load(open(output_dir + str(last_iteration) + '/params.pkl', 'rb'))
            pbar_range = range(last_iteration, last_iteration + iterations)

        # Log files
        shutil.copyfile('experiment.ini', output_dir + 'experiment.ini')
        shutil.copyfile('train.py', output_dir + 'train.py')
        shutil.copyfile('flow_utils.py', output_dir + 'flow_utils.py')

    # Create optimizer
    scheduler = utils.get_scheduler(lr, lr_schedule)
    opt_init, opt_update, get_params = utils.get_optimizer(optimizer, scheduler, b1, b2)
    opt_state = opt_init(params)

    # best_val_params, best_val_loss = None, None
    best_test_params, best_test_loss, best_test_epsilon = None, None, 0.
    pbar = tqdm(pbar_range)
    """
    grad_l2s = []
    """
    for iteration in pbar:
        batch, X = utils.get_batch(sampling, key, X, minibatch_size, iteration)

        # Possible with Poisson subsampling
        if batch.shape[0] == 0:
            continue

        # Perform model update
        temp_key, key = random.split(key)
        if private:
            opt_state = private_update(temp_key, iteration, opt_state, batch)
        else:
            opt_state = update(temp_key, iteration, opt_state, batch)

        # Log params
        if log_params and ((iteration < 10000 and (iteration - 1) % 200 == 0) or iteration % log_params == 0):
            params = get_params(opt_state)
            utils.log(params, output_dir + str(iteration) + '/')

        """
        params = get_params(opt_state)
        grad_l2 = l2(grad(loss)(params, batch))
        grad_l2s.append(grad_l2)
        """

        # Update progress bar
        if iteration % log_performance == 0:
            """
            print(np.median(grad_l2s))
            """

            # Calculate privacy loss
            epsilon = utils.get_epsilon(
                private, composition, sampling, iteration,
                noise_multiplier, X.shape[0], minibatch_size, delta
            )

            # Calculate losses
            params = get_params(opt_state)
            train_loss = loss(params, X)
            test_loss = loss(params, X_test)

            # Exit if NaN, as all has failed...
            if np.isnan(train_loss).any():
                break

            # Update best test model thus far
            if best_test_loss is None or test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_params = params
                best_test_epsilon = epsilon
                if log_params:
                    utils.dump_obj(best_test_params, output_dir + 'test_params.pkl')

            # Update progress bar
            pbar_text = 'Train: {:.3f} Test: {:.3f} ε: {:.3f} Best Test: {:.3f} Best ε: {:.3f}'.format(
                -train_loss, -test_loss, epsilon, -best_test_loss, best_test_epsilon,
            )

            pbar.set_description(pbar_text)


if __name__ == '__main__':
    config_file = 'experiment.ini' if len(sys.argv) == 1 else sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config['DEFAULT']
    print('Best test loss: {}'.format(main(config)))
