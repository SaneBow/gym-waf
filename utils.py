import os
import pickle
from time import gmtime, strftime

SAVE_DIR = 'trained_model'


def save_model(name, model, history=None):
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    ts = strftime("%m%d%H%M", gmtime())
    model.save(os.path.join(SAVE_DIR, '{}_{}.h5'.format(name, ts)))
    if history:
        with open(os.path.join(SAVE_DIR, '{}_history_{}.pickle'.format(name, ts)), 'wb') as f:
            pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
