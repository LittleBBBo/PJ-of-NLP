import sys
import os
from os.path import dirname

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

PROJECT_DIR = dirname(dirname(os.path.abspath(__file__)))
MYLDA_DIR = os.path.join(PROJECT_DIR, "mylda")
sys.path.append(MYLDA_DIR)

from mylda import LDA, Dictionary


def _tokens2words(dictionary, topWordTokens):
    return [dictionary.id2token(x) for x in topWordTokens]


def token2words(dictionary, topWordTokens):
    if hasattr(topWordTokens[0], "__iter__"):
        words = [_tokens2words(dictionary, x) for x in topWordTokens]
        return words
    else:
        words = _tokens2words(dictionary, topWordTokens)
        return words


def _top_words(sub_n_kt, topNum):
    K, T = sub_n_kt.shape
    topWordTokens = [[] for x in range(K)]
    for k in range(K):
        topWordTokens[k] = sub_n_kt[k, :].argsort()[::-1][:topNum]
    return topWordTokens


def m1(dataground, subK=3, topNum=15):
    """return corpus-level topic ordered by
    corresponding proportion in subset"""
    lda_model = dataground.lda_model
    subset_n_mk = dataground.subset_n_mk()
    k_weight = subset_n_mk.sum(axis=0)
    topK_topics = k_weight.argsort()[::-1][:subK]
    sub_n_kt = lda_model.n_kt[topK_topics]
    topWordTokens = _top_words(sub_n_kt, topNum)
    return topWordTokens


def vl(dataground, subK=3, topNum=15):
    """Retrain a LDA model for subset"""
    w_mi = dataground.subset_w_mi()
    new_lda_model = LDA(K=subK, dictionary=dataground.dictionary)
    new_lda_model.fit(w_mi)
    sub_n_kt = new_lda_model.n_kt
    topWordTokens = _top_words(sub_n_kt, topNum)
    return topWordTokens


REPR = dict()
REPR["m1"] = m1
REPR["vl"] = vl
