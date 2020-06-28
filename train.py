import numpy as np
import time
import logging
import json
import os
from argparse import ArgumentParser
import tensorflow as tf
from sklearn.utils import shuffle
from collections import Counter

from models.cnn import NormalCNN, LatentFactorCNN, DoubleLatentCNN, LatentBiasCNN, LowerLatentFactorCNN, LatentWeightCNN, LatentFactorCNN2, LatentWeightOnlyCNN, OneHotCNN, MyLatentWeightCNN, MAPFullMultilevelCNN, MAPFactoredMultilevelCNN
from models.baselines import MLP, OneHotMLP, MultilevelMLP, FactoredMultilevelMLP
from models.lstm import NormalLSTM, LatentFactorLSTM, DoubleLatentLSTM
from utils import set_logger

# Silence sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# filter numpy warnings
warnings.filterwarnings('ignore')

logging.getLogger("tensorflow").setLevel(logging.ERROR)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--lr',
        type=float,
        default=0.004)
    parser.add_argument('--lr-sched',
        action='store_true')
    parser.add_argument('--decay-steps',
        type=int,
        default=100)
    parser.add_argument('--decay-rate',
        type=float,
        default=0.96)
    parser.add_argument('--batch-size',
        help='use larger epochs when using GPU e.g. 1000',
        type=int,
        default=100)
    parser.add_argument('--num-epochs',
        type=int,
        default=1)
    parser.add_argument('--data-dir',
        help='directory with dataset folders',
        type=str,
        default='data')
    parser.add_argument('--dataset',
        help='name of dataset',
        type=str,
        default=None)
    parser.add_argument('--data-size',
        help='pick small or large version of dataset',
        type=str,
        choices=['small', 'large'])
    parser.add_argument('--model-size',
        help='use small or large model parameters',
        type=str,
        default='small')
    parser.add_argument('--latent-config',
        help=('\
        Possible latent configurations for --latent-config:\n\
            "none" : no latent variables\n\
            "bias" : each output node gets a latent bias\n\
            "factor" : append a latent vector to the layer before softmax\n\
            "double" : two latent vector on last layer and second to last\n\
        '),
        type=str)
    # TODO: try to make this more intuitive
    # nargs input format e.g. `--z-dim 10 20` this will be parsed as [10,20]
    parser.add_argument('--z-dim',
        help='number of groups in each group level',
        nargs='+',
        type=int,
        default=1)
    parser.add_argument('--eval-every',
        help='how many epochs in between evaluating on test set',
        type=int,
        default=1)
    parser.add_argument('--print-freq',
        help='how many times progress is printed per epoch',
        type=int,
        default=10)
    parser.add_argument('--seed',
        type=int,
        default=None)
    parser.add_argument('--testing',
        help='write to experiments/tests to keep main exp_dir clean',
        action='store_true')
    parser.add_argument('--description',
        help='add memorable note for context',
        type=str,
        default=None)

    return parser.parse_args()


def make_experiment_directory(data_dir, dataset, testing):
    """
    Experiments are logged in the experiments directory,
    then further sorted by dataset and time.

    TODO: give option to change experiment_dir

    Args:
        dataset (str) : name of dataset
        testing (bool) : whether this is a test run

    --- Example ---
    experiments/
        femnist/
            2019-07-29_18-21-12/
                hyperparams.json
                training_stats.csv
                train.log
                model_weights.pickle
    """
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join('experiments', dataset, timestr)
    # To avoid cluttering experiment directory while testing scripts
    if testing:
        experiment_dir = os.path.join('experiments', 'tests', timestr)
    os.makedirs(experiment_dir)
    return experiment_dir


def read_femnist_data(data_dir, data_size, seed):
    """Read in dataset arrays.

    Args:
        data_size (str) : request either the small
            or large dataset

    Returns a two lists of arrays, for training and testing. The arrays should be of the form [x, gid, gid2, ..., y]. This allows
    decomposition into `inputs, labels = arr[:-1], arr[-1]`
    """


    data_dir = os.path.join(data_dir, 'femnist', data_size)
    load = lambda name: np.load(os.path.join(data_dir, name))

    x_train = load('x_train.npy')
    gid_train = load('gid_train.npy')
    y_train = load('y_train.npy')
    x_test = load('x_test.npy')
    gid_test = load('gid_test.npy')
    y_test = load('y_test.npy')

    train_data = [x_train, gid_train, y_train]
    test_data = [x_test, gid_test, y_test]

    train_data = shuffle(*train_data, random_state=seed)
    test_data = shuffle(*test_data, random_state=seed)
    return train_data, test_data


def read_shakespeare_data(data_dir, data_size, seed):
    """Read in dataset arrays.

    Args:
        data_size (str) : request either the small
            or large dataset

    Returns a two lists of arrays, for training and testing. The arrays should be of the form [x, gid, gid2, ..., y]. This allows
    decomposition into `inputs, labels = arr[:-1], arr[-1]`
    """

    data_dir = os.path.join(data_dir, 'shakespeare_data', data_size)
    load = lambda name: np.load(os.path.join(data_dir, name))

    x_train = load('x_train.npy')
    gid_train = load('gid_train.npy')
    gid2_train = load('gid2_train.npy')
    y_train = load('y_train.npy')
    x_test = load('x_test.npy')
    gid_test = load('gid_test.npy')
    gid2_test = load('gid2_test.npy')
    y_test = load('y_test.npy')

    train_data = [x_train, gid_train, gid2_train, y_train]
    test_data = [x_test, gid_test, gid2_test, y_test]

    train_data = shuffle(*train_data, random_state=seed)
    test_data = shuffle(*test_data, random_state=seed)
    return train_data, test_data

def main():
    start = time.time()

    args = parse_args()

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    experiment_dir = make_experiment_directory(args.data_dir, args.dataset, args.testing)
    logger = set_logger(os.path.join(experiment_dir, 'train.log'))
    with open(os.path.join(experiment_dir, 'hyperparams.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    if args.lr_sched:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            staircase=True)
    else:
        lr = args.lr

    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # clipvalue=2.
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, )
    #import tensorflow_addons as tfa
    #optimizer = tfa.optimizers.LazyAdam(learning_rate=args.lr)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)

    # Tensorflow by default sums over losses
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    kwargs = {
        'optimizer' : optimizer,
        'loss_fn' : loss_fn,
        'args' : args,
        'experiment_dir' : experiment_dir,
        'logger' : logger
    }

    # TODO: possibly move this functionality into the read call,
    # i.e. return data and model
    if args.dataset=='femnist':
        print('>>> Reading FEMNIST...')
        train_data, test_data = read_femnist_data(
            args.data_dir, args.data_size, args.seed)

        model_dict = {
            'none' : NormalCNN,
            'factor' : LatentFactorCNN,
            'double' : DoubleLatentCNN,
            'bias' : LatentBiasCNN,
            'lower' : LowerLatentFactorCNN,
            'weight' : LatentWeightCNN,
            'factor2' : LatentFactorCNN2,
            #'weight-factor' : FactoredWeightCNN,
            'weight-only' : LatentWeightOnlyCNN,
            'mlp' : MLP, 'one-hot': OneHotMLP, 'ml-mlp': MultilevelMLP, 'fml-mlp':FactoredMultilevelMLP,
            'one-hot-cnn' : OneHotCNN,
            'my-ml-dense' : MyLatentWeightCNN,
            'map-full' : MAPFullMultilevelCNN,
            'map-factored' : MAPFactoredMultilevelCNN
        }

    else:
        print('>>> Reading Shakespeare (not literally)...')
        train_data, test_data = read_shakespeare_data(
            args.data_dir, args.data_size, args.seed)
        gid_train = train_data[1]
        gid2_train = train_data[2]
        num_groups = (
            len(np.unique(gid_train)),
            len(np.unique(gid2_train)))
        kwargs['num_groups'] = num_groups

        # TODO: update to dict structure
        if args.latent_config=='none':
            model = NormalLSTM(**kwargs)
        elif args.latent_config=='factor':
            model = LatentFactorLSTM(**kwargs)
        elif args.latent_config=='double':
            model = DoubleLatentLSTM(**kwargs)
        # elif args.latent_config=='bias':
        #     model = LatentBiasLSTM(**kwargs)
        else:
            raise ValueError('No matching latent variable configuration')

    # TODO: should be a cleaner way to get number of unique groups
    _, gid_train, _ = train_data
    # num_groups should be provided as an array-like, even if only one elt
    num_groups = (len(np.unique(gid_train)),)
    train_size = gid_train.shape[0]
    kwargs['train_size'] = train_size
    kwargs['num_groups'] = num_groups

    # Used later for KL scaling, group_train_sizes[i] gives 
    # number of datapoints for group i
    count_dict = Counter(gid_train)
    kwargs['group_train_sizes'] = [count_dict[i] for i in range(num_groups[0])]

    model = model_dict[args.latent_config](**kwargs)

    model.train(train_data, test_data, args.batch_size, args.num_epochs, args.eval_every, args.print_freq)

    end = time.time()
    logger.info('Finished in {}s'.format(round(end-start)))


if __name__ == '__main__':
    main()

