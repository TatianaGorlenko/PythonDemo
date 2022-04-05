from collections import OrderedDict, defaultdict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        tokens_dict = defaultdict(int)
        for row in X:
            for token in row.split():
                tokens_dict[token] += 1
        self.bow = sorted(tokens_dict, key=lambda x: -tokens_dict[x])[: self.k]
        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """
        
        tokens = text.split()
        result = [0] * self.k
        for token in tokens:
            if token in self.bow:
                result[self.bow.index(token)] += 1
        
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        idf_dict = defaultdict(int)
        for text in X:
            for token in set(text.split()):
                idf_dict[token] += 1
        
        most_frequent_tokens = sorted(idf_dict, key=lambda x: -idf_dict[x])[: self.k]
        len_x = len(X)
        self.idf = {k: np.log(len_x / v + 1) for k, v in idf_dict.items() if k in most_frequent_tokens}

        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """

        result = np.zeros(len(self.idf.keys()), dtype=np.float32)
        token_dict = defaultdict(int)
        for token in text.split():
            if token in self.idf.keys():
                token_dict[token] += 1
        for i, token in enumerate(self.idf.keys()):
            if token in token_dict:
                result[i] = token_dict[token] * self.idf[token]
            else:
                result[i] = 0
        if self.normalize:
            result = result / (np.linalg.norm(result, ord=2) + 1e-8)
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
