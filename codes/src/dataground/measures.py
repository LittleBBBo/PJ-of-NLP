from random import sample

import numpy as np
import pandas as pd
try:
    from .subset_repr import REPR
except:
    pass


def _doc_freq_w(wordToken, subset_w_mi):
    count = 0
    for w in subset_w_mi:
        count += (wordToken in w)
    return count


def _co_doc_freq_w(wToken1, wToken2, subset_w_mi):
    count = 0
    for w in subset_w_mi:
        count += (wToken1 in w and wToken2 in w)
    return count


def _pmi_measure_single_topic(topWordTokens, subset_w_mi):
    twt = topWordTokens
    m = len(twt)
    pmi = 0
    for j in range(1, m):
        count_j = _doc_freq_w(twt[j], subset_w_mi)
        for i in range(j):
            count_i = _doc_freq_w(twt[i], subset_w_mi)
            count_ij = _co_doc_freq_w(twt[i], twt[j], subset_w_mi)
            pmi += np.log((count_ij + 1) / ((count_i + 1) * (count_j + 1)))
    return pmi


def _npmi_measure_single_topic(topWordTokens, subset_w_mi):
    twt = topWordTokens
    m = len(twt)
    npmi = 0
    for j in range(1, m):
        count_j = _doc_freq_w(twt[j], subset_w_mi)
        for i in range(j):
            count_i = _doc_freq_w(twt[i], subset_w_mi)
            count_ij = _co_doc_freq_w(twt[i], twt[j], subset_w_mi)
            added = np.log((count_ij + 1) / ((count_i + 1) * (count_j + 1)))
            added /= (-np.log(count_ij + 1))
            npmi += added
    return npmi


def _lcp_measure_single_topic(topWordTokens, subset_w_mi):
    twt = topWordTokens
    m = len(twt)
    lcp = 0
    for j in range(1, m):
        count_j = _doc_freq_w(twt[j], subset_w_mi)
        for i in range(j):
            count_ij = _co_doc_freq_w(twt[i], twt[j], subset_w_mi)
            lcp += np.log((count_ij + 1) / (count_j + 1))
    return lcp


def pmi_measure(topWordTokens, subset_w_mi):
    # if isinstance(topWordTokens[0], list):
    if hasattr(topWordTokens[0], "__iter__"):
        pmi_values = [_pmi_measure_single_topic(row, subset_w_mi)
                      for row in topWordTokens]
        return np.mean(pmi_values)
    else:
        return _pmi_measure_single_topic(topWordTokens, subset_w_mi)


def npmi_measure(topWordTokens, subset_w_mi):
    # if isinstance(topWordTokens[0], list):
    if hasattr(topWordTokens[0], "__iter__"):
        npmi_values = [_npmi_measure_single_topic(row, subset_w_mi)
                       for row in topWordTokens]
        return np.mean(npmi_values)
    else:
        return _npmi_measure_single_topic(topWordTokens, subset_w_mi)


def lcp_measure(topWordTokens, subset_w_mi):
    if hasattr(topWordTokens[0], "__iter__"):
        lcp_values = [_lcp_measure_single_topic(row, subset_w_mi)
                      for row in topWordTokens]
        return np.mean(lcp_values)
    else:
        return _lcp_measure_single_topic(topWordTokens, subset_w_mi)


def _eval_method(dataground, target_ids, method, measure, subK, topNum):
    dataground.set_target_subset(target_ids)
    topWordTokens = method(dataground, subK, topNum)
    value = measure(topWordTokens, dataground.subset_w_mi())
    return value


def compare_experiment(dataground, df=None, REPR=REPR,
                       measure=pmi_measure, subK=3,
                       subM=500, topNum=15,
                       sample_times=10):
    if df is None:
        df = pd.DataFrame(index=["vl", "m1", "m3", "m5"])
    current_col_name = "subset%d_subK%d" % (subM, subK)
    selected = [sample(range(dataground.lda_model.M), subM)
                for _ in range(sample_times)]
    # selected = [sample(range(11314), subM)
    #             for _ in range(sample_times)]
    # selected = [sample(range(11314, dataground.lda_model.M), subM)
    #             for _ in range(sample_times)]
    for method_name in df.index:
        method = REPR[method_name]
        values = [_eval_method(dataground, target_ids, method,
                               measure, subK, topNum)
                  for target_ids in selected]
        avg_value = np.mean(values)
        df.ix[method_name, current_col_name] = avg_value
    return df
