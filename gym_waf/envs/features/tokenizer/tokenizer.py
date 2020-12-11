"""
Based on 

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6163109
Classification of Malicious Web Code by Machine Learning - Komiya et al.

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6993127
SQL Injection Detection using Machine Learning

https://www.sciencedirect.com/science/article/pii/S0167404816300451
SQLiGoT: Detecting SQL injection attacks using graph of tokens and SVM

"""
import os
import re
import numpy as np
import sqlparse
import sqlparse.tokens as tks
from collections import OrderedDict
from . import allowed_tokens as alt


class TokenizerType:
    """Tokenizer class. Use sqlparse library. Produce short feature vector (12) on token-type level. """

    def __init__(self):
        self._allowed_tokens = [
            tks.Other,
            tks.Keyword,
            tks.Name,
            tks.String,
            tks.Number,
            tks.Punctuation,
            tks.Operator,
            tks.Comparison,
            tks.Wildcard,
            tks.Comment.Single,
            tks.Comment.Multiline,
            tks.Operator.Logical,
        ]
        self.vect_size = len(self._allowed_tokens)

    def get_allowed_tokens(self):
        """Returns the tokens used for creating the feature vector.
        
        Returns:
            [list] : list containing all the tokens.
        """
        return self._allowed_tokens

    def _produce_tokens(self, parsed: list):
        """Given a list of sql-parse tokens, it returns a list of only the type of each token.
        
        Arguments:
            parsed (list) : The sql-parse output
        
        Returns:
            list : List of tokens
        """
        resulting_tokens = []
        for i in parsed:
            resulting_tokens.append(i.ttype)
        return resulting_tokens

    def produce_feat_vector(self, sql_query: str, normalize=False):
        """It returns the feature vector as histogram of tokens, produced from the input query.
        
        Arguments:
            sql_query (str) : An input SQL query
        
        Keyword Arguments:
            normalize (bool) : True for producing a normalized hitogram. (default: (False))
        
        Raises:
            TypeError: params has wrong types
        
        Returns:
            numpy ndarray : histogram of tokens
        """

        parsed = list(sqlparse.parse(sql_query)[0].flatten())
        allowed = self._allowed_tokens
        tokens = self._produce_tokens(parsed)
        dict_token = OrderedDict(zip(allowed, [0 for _ in range(len(allowed))]))
        for t in tokens:
            if t in dict_token:
                dict_token[t] += 1
            else:
                parent = t
                while parent is not None and parent not in dict_token:
                    parent = parent.parent
                if parent is None:
                    continue
                dict_token[parent] += 1
        values = dict_token.values()
        feature_vector = np.array([i for i in values])
        if normalize:
            norm = np.linalg.norm(feature_vector)
            feature_vector = feature_vector / norm
        return feature_vector

    def create_dataset_from_file(
        self, filepath: str, label: int, limit: int = None, unique_rows=True
    ):
        """Create dataset from fil containing sql queries.
        
        Arguments:
            filepath (str) : path of sql queries dataset
            label (int) : labels to assign to each sample
        
        Keyword Arguments:
            limit (int) : if None, it specifies how many queries to use (default: (None))
            unique_rows (bool) : True for removing all the duplicates (default: (True))
        
        Raises:
            TypeError: params has wrong types
            FileNotFoundError: filepath not pointing to regular file
            TypeError: limit is not None and not int
        
        Returns:
            (numpy ndarray, list) : X and y
        """

        assert os.path.exists(filepath)
        X = []
        with open(filepath, "r") as f:
            i = 0
            for line in f:
                if limit is not None and i > limit:
                    break
                line = line.strip()
                X.append(self.produce_feat_vector(line))
                i += 1
        if unique_rows:
            X = np.unique(X, axis=0)
        else:
            X = np.array(X)
        y = [label for _ in X]
        return X, y


class TokenizerTK:
    """ TokenizerTK. Use self-defined tokens. Produce long feature vector (702) on token level. """
    def __init__(self):
        self.vect_size = len(alt.TOKENS)

    def produce_feat_vector(self, sql_query: str, normalize=False):
        tokens = self._preprocess_input_query(sql_query)
        token_counts = self._histogram_of_tokens(tokens)
        feature_vector = np.array(token_counts)
        if normalize:
            norm = np.linalg.norm(feature_vector)
            feature_vector = feature_vector / norm
        return feature_vector 

    def _preprocess_input_query(self, query):
        query = query.strip().upper()
        query = re.sub(r"( |\t|\n|\r|/\*\*/|`)+", " ", query)
        query = alt.substitute_sysinfo(query, insert_space=True).strip()
        query = alt.apply_regexp(query, insert_space=True).strip()
        query = alt.substitute_punctation(query, insert_space=True).strip()
        query = re.sub(" +", " ", query).strip()
        query = alt.normalize_dots(query)
        splitted_string = query.split(" ")
        tokens = []
        for t in splitted_string:
            if t in alt.TOKENS:
                tokens.append(t)
            else:
                if len(t) > 1:
                    tokens.append("STR")
                else:
                    tokens.append("CHR")
        if not tokens:
            return None
        return tokens

    def _histogram_of_tokens(self, tokens):
        hist = [0 for _ in range(self.vect_size)]
        for t in tokens:
            hist[alt.TOKENS.index(t)] = tokens.count(t)
        return hist


class TokenizerChr:
    """ TokenizerChr. Produce character histogram feature vector (256). """
    def __init__(self):
        self.vect_size = 0xff

    def produce_feat_vector(self, sql_query: str, normalize=False):
        token_counts = self._histogram_of_chars(sql_query)
        feature_vector = np.array(token_counts)
        if normalize:
            norm = np.linalg.norm(feature_vector)
            feature_vector = feature_vector / norm
        return feature_vector  

    def _histogram_of_chars(self, s):
        hist = [0 for _ in range(0xff)]
        sb = s.encode()
        for c in sb:
            hist[c] = sb.count(c)
        return hist