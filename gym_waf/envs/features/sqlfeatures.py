from .tokenizer import *
import numpy as np

class SqlFeatureExtractor(object):

    def __init__(self):
        pass

    def extract(self, payload):
        # feature vectors that require only raw bytez
        tokenizer_type = TokenizerType()  # this generates a 12 dimension feature vector
        tokenizer_tk = TokenizerTK()      # this generates a 702 dimension feature vector
        tokenizer_chr = TokenizerChr()    # this generates a 256 dimension feature vector
        feature_type = tokenizer_type.produce_feat_vector(payload)
        feature_tk = tokenizer_tk.produce_feat_vector(payload)
        feature_chr = tokenizer_chr.produce_feat_vector(payload)
        # feature_vector = np.concatenate((feature_type, feature_tk, feature_chr))
        feature_vector = np.concatenate((feature_type, feature_chr))

        return feature_vector