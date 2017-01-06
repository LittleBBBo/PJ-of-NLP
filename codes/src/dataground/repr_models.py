import sys
import os
import time
from os.path import dirname
from random import sample

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

PROJECT_DIR = dirname(dirname(os.path.abspath(__file__)))
MYLDA_DIR = os.path.join(PROJECT_DIR, "mylda")
sys.path.append(MYLDA_DIR)

from mylda import LDA, Dictionary, _sample
import ac_infant
# from .distanceFactory import DIST_METHOD
# _distance_method = DIST_METHOD["default"]
_distance_method = None


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


def _distance(pA, pB):
    # if _distance_method is not None:
        # return _distance_method(pA, pB)
    smooth = 0.1
    pA = pA + smooth
    pB = pB + smooth
    pA = pA / pA.sum()
    pB = pB / pB.sum()
    H = pA.dot(pB)
    H = np.sqrt(H)
    H = 1 - H.sum()
    return H



def _init_center(waiting_list, n_kt_refined, subK):
    wl = waiting_list.copy()
    K, T = n_kt_refined.shape
    np.random.shuffle(wl)
    selected = wl[:subK]
    centers = np.empty((subK, T))
    for i in range(subK):
        centers[i] = n_kt_refined[selected[i]]

    # for i in subK:
    # centers_id = [1, 3, 5]
    return centers


def _distance_to_selected(item, selected, n_kt_refined): # compute distance between item and selected set
    result = 0
    for select in selected:
        result += _distance(n_kt_refined[item], n_kt_refined[select])
    return result


def _init_center_method1(waiting_list, n_kt_refined, subK): # better version to choose random centers
    wl = waiting_list.copy()
    K, T = n_kt_refined.shape
    selected = []
    selected.append(np.random.choice(wl))
    for i in range(subK - 1):
        distance = -1
        new_selec = 0
        for item in wl:
            if item not in selected:
                distance_tmp = _distance_to_selected(item, selected, n_kt_refined)
                if distance_tmp > distance:
                    distance = distance_tmp
                    new_selec = item
        selected.append(new_selec)

    centers = np.empty((subK, T))
    for i in range(subK):
        centers[i] = n_kt_refined[selected[i]]

    # for i in subK:
    # centers_id = [1, 3, 5]
    return centers


def _init_center_method2(waiting_list, n_kt_refined, subK):
    K, T = n_kt_refined.shape
    k_weight = n_kt_refined.sum(axis=1)
    topK_topics = k_weight.argsort()[::-1]
    centers = np.empty((subK, T))
    for i in range(subK):
        centers[i] = n_kt_refined[topK_topics[i]]
    return centers


def _update_centers(centers, n_kt_refined, waiting_list):
    subK = len(centers)
    K, T = n_kt_refined.shape
    cluster_ids = [[] for k in range(subK)]
    new_centers = np.zeros((subK, T))
    for k in waiting_list:
        distances = [_distance(n_kt_refined[k], center)
                     for center in centers]
        category_to = distances.index(min(distances))
        cluster_ids[category_to].append(k)
        new_centers[category_to] += n_kt_refined[k]
    return (new_centers, cluster_ids)


def _infant_cluster(n_kt_refined, subK):
    K, T = n_kt_refined.shape
    waiting_list = []
    for k in range(K):
        if n_kt_refined[k, :].sum() > 0:
            waiting_list.append(k)
    centers = _init_center_method2(waiting_list, n_kt_refined, subK)
    iter_max = 50
    for i in range(iter_max):
        (new_centers, cluster_ids) = _update_centers(centers,
                                                     n_kt_refined,
                                                     waiting_list)
        delta = new_centers - centers
        delta = abs(delta).sum()
        print("change: %d" % delta)
        centers = new_centers
        if delta == 0:
            break
    return centers


class ReprModel:
    """topic model for subset"""
    def __init__(self, n_kt, alpha=0.1, beta=0.1):
        self.n_kt = n_kt
        self.alpha = alpha
        self.beta = beta
        self.K, self.T = n_kt.shape
        n_kt = n_kt + beta
        self.phi_kt = n_kt / np.c_[n_kt.sum(axis=1)]

    def _sample_word(self, n_z, w_i):
        pval = np.zeros(self.K)
        for k in range(self.K):
            pval[k] = self.phi_kt[k, w_i] * (self.alpha + n_z[k])
        pval = pval / pval.sum()
        z = np.random.multinomial(1, pval)
        z = z.nonzero()[0][0]
        return z

    def _sample_doc(self, n_z, z, doc):
        n = len(doc)
        if n <= 1:
            raise Exception("doc is too short")
        for i in range(n):
            z_i = z[i]
            n_z[z_i] -= 1
            new_z = self._sample_word(n_z, doc[i])
            z[i] = new_z
            n_z[new_z] += 1

    def predict(self, doc, iter_max=50):
        try:
            [int(wi) for wi in doc]
        except:
            raise Exception("doc should be a sequence of tokens")
        n = len(doc)
        z = np.random.randint(0, self.K, n, dtype=np.intc)
        n_z = np.zeros(self.K)
        for z_i in z:
            n_z[z_i] += 1
        for i in range(iter_max):
            self._sample_doc(n_z, z, doc)
        return z

    def _predict(self, doc, iter_max=50):
        try:
            [int(wi) for wi in doc]
            doc = np.array(doc)
        except:
            raise Exception("doc should be a sequence of tokens")
        n = len(doc)
        z = np.random.randint(0, self.K, n, dtype=np.intc)
        n_z = np.zeros(self.K, dtype=np.intc)
        for z_i in z:
            n_z[z_i] += 1
        for i in range(iter_max):
            _sample._predict(self.phi_kt, doc, z, n_z, self.alpha)
        return z

    def _approximate_likelihood(self, doc, z=None):
        # # old version
        # ll = 0
        # n = len(doc)
        # if z is None:
        #     z = self._predict(doc)
        # for i in range(n):
        #     ll += np.log(self.phi_kt[z[i], doc[i]])
        # return ll
        ll = 0
        n = len(doc)
        nz = np.zeros(self.K)
        if z is None:
            z = self._predict(doc)
        for i in range(n):
            ll += np.log(self.phi_kt[z[i], doc[i]])
            nz[z[i]] += 1
        ll += np.log(self.alpha * self.K) - np.log(self.alpha * self.K + n)
        for k in range(self.K):
            if nz[k] > 0:
                ll += np.log(self.alpha + nz[k]) - np.log(self.alpha)
        # return ll
        if n == 0:
            return 0
        avg_ll = ll / n
        return avg_ll

    def approximate_likelihood(self, docs):
        if hasattr(docs[0], "__iter__"):
            values = [self._approximate_likelihood(doc) for doc in docs]
            # return np.sum(values)
            return np.mean(values)
        else:
            return self._approximate_likelihood(docs)

    def llh(self, doc, z):
        """calculate complete loglikelihood, log p(w, z)

        Formula used is log p(w, z) = log p(w|z) + log p(z)
        """
        pass

    def top_words(self, topNum=15):
        topWordTokens = _top_words(self.n_kt, topNum)
        return topWordTokens


def m1(dataground, subK=5):
    """retain top-proportion corpus-level topic, and assign other words with
    'others' topic token
    """
    lda_model = dataground.lda_model
    # subset_n_mk = dataground.subset_n_mk_train()
    subset_n_mk = dataground.subset_n_mk()  # use same subset to train and test
    k_weight = subset_n_mk.sum(axis=0)
    # topK_topics = k_weight.argsort()[::-1][:subK]
    # sub_n_kt = lda_model.n_kt[topK_topics]
    # return ReprModel(sub_n_kt)
    index_sorted_weight = k_weight.argsort()[::-1]
    topK_1_topics = index_sorted_weight[:subK - 1] # top k - 1 topics
    sub_n_kt = lda_model.n_kt[topK_1_topics]

    remain_topics = index_sorted_weight[subK:]
    remain_sub_n_kt = lda_model.n_kt[remain_topics].copy()
    sum_weight = 0
    for i in range(len(remain_topics)):
        k = remain_topics[i]
        sum_weight += k_weight[k]
        for item in remain_sub_n_kt[i]:
            item *= k_weight[k]
    remain_sub_n_kt = remain_sub_n_kt.sum(axis=0)
    for item in remain_sub_n_kt:
        item = int(item / sum_weight)
    result = np.concatenate((sub_n_kt, np.array([remain_sub_n_kt])), axis=0)
    return ReprModel(result)


def vl(dataground, subK=5):
    # w_mi_train = dataground.subset_w_mi_train()
    w_mi_train = dataground.subset_w_mi()  # use same subset to train and test
    new_lda_model = LDA(K=subK, dictionary=dataground.dictionary)
    new_lda_model.fit(w_mi_train)
    sub_n_kt = new_lda_model.n_kt
    return ReprModel(sub_n_kt)


def infant(dataground, subK=5):
    K, T = dataground.lda_model.n_kt.shape
    # w_mi = dataground.subset_w_mi_train()
    # z_mi = dataground.subset_z_mi_train()
    w_mi = dataground.subset_w_mi()  # use same subset to train and test
    z_mi = dataground.subset_z_mi()  # use same subset to train and test
    M = len(w_mi)
    n_kt_refined = np.zeros((K, T), dtype=np.intc)
    for m in range(M):
        for (z, w) in zip(z_mi[m], w_mi[m]):
            n_kt_refined[z, w] += 1
    sub_n_kt = _infant_cluster(n_kt_refined, subK)
    return ReprModel(sub_n_kt)


def infant_acd(dataground, subK=5):
    K, T = dataground.lda_model.n_kt.shape
    # w_mi = dataground.subset_w_mi_train()
    # z_mi = dataground.subset_z_mi_train()
    w_mi = dataground.subset_w_mi()  # use same subset to train and test
    z_mi = dataground.subset_z_mi()  # use same subset to train and test
    M = len(w_mi)
    n_kt_refined = np.zeros((K, T), dtype=np.intc)
    for m in range(M):
        ac_infant._count_n_kt_refined(n_kt_refined, z_mi[m], w_mi[m])
        # for (z, w) in zip(z_mi[m], w_mi[m]):
        # n_kt_refined[z, w] += 1
    sub_n_kt = ac_infant._infant_cluster(n_kt_refined, subK)
    return ReprModel(sub_n_kt)


REPR = dict()
REPR["m1"] = m1
REPR["vl"] = vl
# REPR["infant"] = infant
REPR["infant"] = infant_acd

def _eval_method(dataground, target_ids, method, subK):
    dataground.set_target_subset(target_ids)
    start = time.time()
    repr_model = method(dataground, subK)
    end = time.time()
    print(end - start)
    # test_docs = dataground.subset_w_mi_test()
    test_docs = dataground.subset_w_mi()  # use same subset to train and test
    return repr_model.approximate_likelihood(test_docs)


def _eval_method_time(dataground, target_ids, method, subK):
    dataground.set_target_subset(target_ids)
    start = time.clock()
    repr_model = method(dataground, subK)
    end = time.clock()
    # print(end - start)
    return end - start


def compare_experiment(dataground, df=None, REPR=REPR, current_col_name=None,
                       subK=3, subM=500, sample_times=10, eval_method=_eval_method):
    if df is None:
        # df = pd.DataFrame(index=["vl", "m1", "m3", "m5"])
        df = pd.DataFrame(index=list(REPR.keys()))
    if current_col_name is None:
        current_col_name = "subM%d_subK%d" % (subM, subK)
    selected = [sample(range(dataground.lda_model.M), subM)
                for _ in range(sample_times)]
    for method_name in df.index:
        method = REPR[method_name]
        values = [eval_method(dataground, target_ids, method, subK)        for target_ids in selected]
        avg_value = np.mean(values)
        df.ix[method_name, current_col_name] = avg_value
    return df


def compare_distance_method(dataground, df=None,
                            REPR=REPR, DIST_METHOD=None,
                            subK=3, subM=500, sample_times=10):
    if DIST_METHOD is None:
        return
    global _distance_method
    for item in DIST_METHOD.items():
        name = "dist_%s" % (item[0])
        _distance_method = item[1]
        df = compare_experiment(dataground=dataground, df=df, REPR=REPR, current_col_name=name,
                           subK=subK, subM=subM, sample_times=sample_times)
        print(df)
    return df


def compare_begin(dataground, cases='default', REPR=REPR, eval_method=_eval_method, time_test=False):
    test_cases00 = [[200, 3], [500, 3]]
    test_cases0 = [[200, 3], [500, 3], [500, 5], [1000, 5], [1000, 10]]
    test_cases1 = [[500, 3], [500, 5], [500, 7], [500, 9], [500, 12], [500, 16], [500, 20]]
    test_cases2 = [[1000, 3], [1000, 5], [1000, 7], [1000, 9], [1000, 12], [1000, 16], [1000, 20]]
    test_cases3 = [[100, 7], [300, 7], [500, 7], [700, 7], [1000, 7], [1500, 7], [2000, 7]]
    test_cases4 = [[100, 9], [300, 9], [500, 9], [700, 9], [1000, 9], [1500, 9], [2000, 9]]
    test_cases5 = [[500, 3], [500, 5], [500, 7], [500, 9], [500, 10], [500, 12], [500, 14], [500, 16], [500, 18], [500, 20]]
    test_cases6 = [[1000, 3], [1000, 5], [1000, 7], [1000, 9], [1000, 10], [1000, 12], [1000, 14], [1000, 16], [1000, 18], [1000, 20]]
    test_cases7 = [[200, 7], [400, 7], [600, 7], [800, 7], [1000, 7], [1200, 7], [1400, 7], [1600, 7], [1800, 7], [2000, 7]]
    test_cases8 = [[200, 9], [400, 9], [600, 9], [800, 9], [1000, 9], [1200, 9], [1400, 9], [1600, 9], [1800, 9], [2000, 9]]
    cases_dict = dict()
    cases_dict['test'] = test_cases00
    cases_dict['default'] = test_cases0
    cases_dict['M500'] = test_cases1
    cases_dict['M1000'] = test_cases2
    cases_dict['K7'] = test_cases3
    cases_dict['K9'] = test_cases4
    cases_dict['M500_T'] = test_cases5
    cases_dict['M1000_T'] = test_cases6
    cases_dict['K7_T'] = test_cases7
    cases_dict['K9_T'] = test_cases8
    print("ready to test!")

    if cases_dict.get(cases) is None:
        print("cases error!")
        return None
    test_cases = cases_dict[cases]

    df = None
    for case in test_cases:
        subM = case[0]
        subK = case[1]
        df = compare_experiment(dataground, df, subK=subK, subM=subM, REPR=REPR, eval_method=eval_method)
    return df


REPR2 = dict()
REPR2["vl"] = vl
# REPR2["infant"] = infant
REPR2["infant"] = infant_acd


def compare_begin_time(dataground, cases='default'):
    return compare_begin(dataground, cases=cases, REPR=REPR2, eval_method=_eval_method_time, time_test=True)

print('better infant!')