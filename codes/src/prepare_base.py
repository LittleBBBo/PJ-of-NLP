from mylda.mylda import LDA, Dictionary
from mylda.mylda.settings import dict_file as default_dict


class DataGround:
    """DataGround is used to store the corpus-level information"""
    def __init__(self, K=100, n_early_stop=30, use_default_dict=False,
                 dict_file=None, min_tf=5, min_df=2, max_dict_len=8000,
                 stem=False):
        self.K = K
        self.n_early_stop = n_early_stop
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
        self.target_subset_ids = range(int(self.lda_model.M))

    def set_target_subset(self, selected_ids):
        self.target_subset_ids = selected_ids

    def subset_w_mi(self, with_dictionary=False):
        subset_w_mi = [self.lda_model.w_mi[x] for x in self.target_subset_ids]
        if with_dictionary:
            return (subset_w_mi, self.dictionary)
        else:
            return subset_w_mi

    def subset_z_mi(self):
        lda_model = self.lda_model
        z_mi = []
        ids = self.target_subset_ids
        for doc_id in ids:
            start = lda_model.I_m[doc_id]
            end = start + lda_model.N_m[doc_id]
            z_mi.append(lda_model.Z[start:end])
        return z_mi
