from .tokenizer import Tokenizer

class SqlFeatureExtractor(object):

    def __init__(self):
        pass

    def extract(self, payload):
        # feature vectors that require only raw bytez
        tokenizer = Tokenizer()
        feature_vector = tokenizer.produce_feat_vector(payload)

        return feature_vector