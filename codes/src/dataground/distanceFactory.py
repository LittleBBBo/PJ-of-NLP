import numpy as np
import scipy.spatial.distance as dist

#test distance method
def pdist(pA, pB):
    return


def smooth_vector(pA, pB):
    smooth = 0.1
    pA = pA + smooth
    pB = pB + smooth
    pA = pA / pA.sum()
    pB = pB / pB.sum()
    return pA, pB


def distance_default(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    H = pA.dot(pB)
    H = np.sqrt(H)
    H = 1 - H.sum()
    return H


def distance_euclidean(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    # return np.sqrt(np.sum((pA - pB) ** 2))
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'euclidean')[0]


def distance_seuclidean(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'seuclidean')[0]


def distance_manhattan(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    # return (abs(pA - pB)).sum()
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'cityblock')[0]


def distance_chebyshev(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    # return abs(pA - pB).max()
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'chebyshev')[0]


def distance_cosine(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    # return 1 - (np.dot(pA, pB) / (np.linalg.norm(pA) * np.linalg.norm(pB)))
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'cosine')[0]


def distance_jaccard(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'jaccard')[0]


def distance_bravcurtis(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'braycurtis')[0]


def distance_canberra(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'canberra')[0]


def distance_correlation(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'correlation')[0]


def distance_dice(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'dice')[0]


def distance_hamming(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'hamming')[0]


def distance_kulsinski(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'kulsinski')[0]


def distance_mahalanobis(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'mahalanobis')[0]


def distance_matching(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'matching')[0]


def distance_minkowski(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'minkowski')[0]


def distance_rogerstanimoto(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'rogerstanimoto')[0]


def distance_russellrao(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'russellrao')[0]


def distance_sokalmichener(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'sokalmichener')[0]


def distance_sokalsneath(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'sokalsneath')[0]


def distance_sqeuclidean(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'sqeuclidean')[0]


def distance_yule(pA, pB, isSmooth = True):
    if isSmooth:
        pA, pB = smooth_vector(pA, pB)
    matV = np.mat([pA, pB])
    return dist.pdist(matV, 'yule')[0]


DIST_METHOD = dict()
DIST_METHOD["default"] = distance_default
DIST_METHOD["braycurtis"] = distance_bravcurtis
DIST_METHOD["canberra"] = distance_canberra
DIST_METHOD["chebyshev"] = distance_chebyshev
DIST_METHOD["cityblock"] = distance_manhattan
DIST_METHOD["correlation"] = distance_correlation
DIST_METHOD["cosine"] = distance_cosine
DIST_METHOD["dice"] = distance_dice
DIST_METHOD["euclidean"] = distance_euclidean
DIST_METHOD["hamming"] = distance_hamming
DIST_METHOD["jaccard"] = distance_jaccard
DIST_METHOD["kulsinski"] = distance_kulsinski
# DIST_METHOD["mahalanobis"] = distance_mahalanobis
DIST_METHOD["matching"] = distance_minkowski
DIST_METHOD["minkowski"] = distance_minkowski
DIST_METHOD["rogerstanimoto"] = distance_rogerstanimoto
DIST_METHOD["russellrao"] = distance_russellrao
DIST_METHOD["seuclidean"] = distance_seuclidean
DIST_METHOD["sokalmichener"] = distance_sokalmichener
DIST_METHOD["sokalsneath"] = distance_sokalsneath
DIST_METHOD["sqeuclidean"] = distance_sqeuclidean
# DIST_METHOD["yule"] = distance_yule

DIST_METHOD_BACKUP = dict()
DIST_METHOD_BACKUP["mahalanobis"] = distance_mahalanobis  # error
DIST_METHOD_BACKUP["yule"] = distance_yule  # nan



#test
vector1 = 100 * np.random.random(size=18000)
vector2 = 100 * np.random.random(size=18000)
print(vector1)
print(vector2)
for item in DIST_METHOD.items():
    print(item[0] + ": ", end='')
    print(item[1](vector1, vector2))

