import argparse
import os
# import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def my_mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)
        return 1
    return 0

def get_best_epoch(config):
    exp_name = config.init_exp
    x = EventAccumulator(path='../experiments/%s/summary/dev'%exp_name)
    x.Reload()
    _, _, vals = zip(*x.Scalars('ccc_%s_1'%config.target_ccc))

    return np.argmax(list(vals))+1