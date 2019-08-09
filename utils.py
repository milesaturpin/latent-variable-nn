import logging
import tensorflow as tf
import numpy as np
import sys
#from prefetch_generator import BackgroundGenerator, background

from sklearn.metrics import log_loss


#@background(max_prefetch=3)
def create_generator(arrays, batch_size):
    data_size = arrays[0].shape[0]
    start = 0
    while start < data_size:
        end = start + batch_size
        yield [array[start:end] for array in arrays]
        start = end


def robust_loss(y_true, y_score, labels=range(62), average=None):
    """Use when have sparse positive labels."""
    try:
        loss = log_loss(y_true, y_score, labels=labels)
    except ValueError:
        loss = np.nan
    return loss


def round_nums(*args, decimal_points=4):
    """Round provided numbers to specified decimal point."""
    return list(map(lambda x: round(x, decimal_points), args))


def set_logger(log_path):
    """
    Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output
    to the terminal is saved in a permanent file.

    Args:
        log_path (string) : where to log
    """
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        # %(levelname)s :
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(stream_handler)

    return logger


def softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.math.log(tf.math.expm1(x))



