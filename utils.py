import os
import pickle
from time import gmtime, strftime

SAVE_DIR = 'results'


def prepare_logdir():
    logdir = os.path.join(SAVE_DIR, strftime("%Y%m%d%H%M", gmtime()))
    os.makedirs(logdir, exist_ok=True)
    return logdir


def save_keras_model(name, model, history=None):
    logdir = prepare_logdir()
    model.save(os.path.join(logdir, '{}.h5'.format(name)))
    if history:
        with open(os.path.join(logdir, '{}_history.pickle'.format(name)), 'wb') as f:
            pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
