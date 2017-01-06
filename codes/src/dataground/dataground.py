import sys
import os
import random
from os.path import dirname

import numpy as np

PROJECT_DIR = dirname(dirname(os.path.abspath(__file__)))
MYLDA_DIR = os.path.join(PROJECT_DIR, "mylda")
sys.path.append(MYLDA_DIR)

from mylda import LDA, Dictionary
from mylda.settings import dict_file as default_dict

# DataGround
class DataGround:
    """DataGround is used to store the corpus-level information"""
    def __init__(self, K=100, n_early_stop=30, use_default_dict=False,
                 dict_file=None, min_tf=5, min_df=2, max_dict_len=8000,
                 stem=False, held_out_ratio=0.2):
        self.K = K
        self.n_early_stop = n_early_stop
        self.held_out_ratio = held_out_ratio
        if use_default_dict:
            self.dictionary = Dictionary(dict_file=default_dict)
        elif dict_file:
            self.dictionary = Dictionary(dict_file=dict_file)
        else:
            self.dictionary = None
            self._min_tf = min_tf
            self._min_df = min_df
            self._max_dict_len = max_dict_len
            self._stem = stem

    def load_data(self, corpus):
        if self.dictionary is not None:
            self.lda_model = LDA(dictionary=self.dictionary,
                                 K=self.K,
                                 n_early_stop=self.n_early_stop)
            self.lda_model.fit(corpus)
        else:
            self.lda_model = LDA(K=self.K, n_early_stop=self.n_early_stop)
            self.lda_model.fit(corpus,
                               min_df=self._min_df,
                               min_tf=self._min_tf,
                               max_dict_len=self._max_dict_len,
                               stem=self._stem)
            self.dictionary = self.lda_model.dictionary
        # self.target_subset_ids = range(int(self.lda_model.M / 20))
        self.set_target_subset(range(int(self.lda_model.M / 20)),
                               random_state=0)

    def _split(self, length, held_out_ratio=0.2, random_state=0):
        train_length = int(length * (1 - held_out_ratio))
        indexes = list(range(length))
        random.seed(random_state)
        random.shuffle(indexes)
        train_index = indexes[:train_length]
        test_index = indexes[train_length:]
        return train_index, test_index

    def set_target_subset(self, selected_ids, random_state=0):
        self.target_subset_ids = selected_ids
        subset_w_mi = self.subset_w_mi()
        indexes = [self._split(len(doc), self.held_out_ratio, random_state)
                   for doc in subset_w_mi]
        self.train_index = [x[0] for x in indexes]
        self.test_index = [x[1] for x in indexes]

    def subset_w_mi(self, with_dictionary=False):
        subset_w_mi = [self.lda_model.w_mi[x] for x in self.target_subset_ids]
        if with_dictionary:
            return (subset_w_mi, self.dictionary)
        else:
            return subset_w_mi

    def subset_w_mi_train(self):
        subset_w_mi = [np.array(doc) for doc in self.subset_w_mi()]
        subset_w_mi_train = [doc[index] for (doc, index)
                             in zip(subset_w_mi, self.train_index)]
        return subset_w_mi_train

    def subset_w_mi_test(self):
        subset_w_mi = [np.array(doc) for doc in self.subset_w_mi()]
        subset_w_mi_test = [doc[index] for (doc, index)
                            in zip(subset_w_mi, self.test_index)]
        return subset_w_mi_test

    def subset_z_mi(self):
        lda_model = self.lda_model
        z_mi = []
        ids = self.target_subset_ids
        for doc_id in ids:
            start = lda_model.I_m[doc_id]
            end = start + lda_model.N_m[doc_id]
            z_mi.append(lda_model.Z[start:end])
        return z_mi

    def subset_z_mi_train(self):
        subset_z_mi = self.subset_z_mi()
        subset_z_mi_train = [z[index] for (z, index)
                             in zip(subset_z_mi, self.train_index)]
        return subset_z_mi_train

    def subset_z_mi_test(self):
        subset_z_mi = self.subset_z_mi()
        subset_z_mi_test = [z[index] for (z, index)
                            in zip(subset_z_mi, self.test_index)]
        return subset_z_mi_test

    def subset_n_mk(self):
        n_mk = self.lda_model.n_mk
        return n_mk[self.target_subset_ids]

    def subset_n_mk_train(self):
        z_mi_train = self.subset_z_mi_train()
        M = len(z_mi_train)
        n_mk_train = np.zeros((M, self.K))
        for m in range(M):
            for z_i in z_mi_train[m]:
                n_mk_train[m, z_i] += 1
        return n_mk_train

    def subset_n_mk_test(self):
        z_mi_test = self.subset_z_mi_test()
        M = len(z_mi_test)
        n_mk_test = np.zeros((M, self.K))
        for m in range(M):
            for z_i in z_mi_test[m]:
                n_mk_test[m, z_i] += 1
        return n_mk_test
