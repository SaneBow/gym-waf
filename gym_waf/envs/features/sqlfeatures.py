from .tokenizer import *
import numpy as np


class SqlFeatureExtractor(object):

    def __init__(self):
        # tokenizer_type = TokenizerType()  # this generates a 12 dimension feature vector
        tokenizer_tk = TokenizerTK()      # this generates a 702 dimension feature vector
        tokenizer_chr = TokenizerChr()    # this generates a 256 dimension feature vector
        self.tokenizers = [tokenizer_tk, tokenizer_chr]
        self.shape = (sum([tknz.vect_size for tknz in self.tokenizers]),)

    def extract(self, payload):
        feature_vector = np.array([])
        for tknz in self.tokenizers:
            new_feat = tknz.produce_feat_vector(payload, normalize=True)
            feature_vector = np.concatenate((feature_vector, new_feat))

        # normalize concatenated vector
        norm = np.linalg.norm(feature_vector)
        feature_vector = feature_vector / norm

        return feature_vector
