# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free, srand, rand, RAND_MAX
import numpy as np
cimport numpy as np




cpdef _count_n_kt_refined(int[:,:] n_kt_refined, int[:] z_mi, int[:] w_mi):
    cdef:
        int w, z, i = 0, n = z_mi.shape[0]

    for i in range(n):
        inc(n_kt_refined[z_mi[i], w_mi[i]])
    return n_kt_refined


cpdef _distance(double[:] pA, double[:] pB):
    # double H
    # H = np.sqrt(pA.dot(pB))
    H = np.sqrt(np.dot(pA, pB))
    return 1 - H
    

cpdef _assign_destination(double[:,:] phi_refined, double[:,:] phi_centers,
                          int[:] cluster_destinations):
    cdef:
        int subk, subk_selected, subK = phi_centers.shape[0]
        int k, K = phi_refined.shape[0]
        double distance, min_distance = 9999

    for k in range(K):
        subk_selected = 0
        min_distance = 9999
        for subk in range(subK):
            distance = _distance(phi_refined[k], phi_centers[subk])
            if distance < min_distance:
                min_distance = distance
                subk_selected = subk
        cluster_destinations[k] = subk_selected


cpdef _update_centers_first(int[:,:] centers, int[:,:] n_kt_refined,
                            int[:] cluster_destinations):
    cdef:
        int k, K = cluster_destinations.shape[0], subK=centers.shape[0]
        int t, T = n_kt_refined.shape[1]

    for k in range(subK):
        centers[k] = 0
    for k in range(K):
        for t in range(T):
            centers[cluster_destinations[k], t] += n_kt_refined[k, t]


cpdef _update_centers(int[:,:] centers, int[:,:] n_kt_refined,
                      int[:] cluster_destinations_old,
                      int[:] cluster_destinations_new):
    
    cdef:
        int K, k, subk_new, subk_old, changed=0
        int t, T = n_kt_refined.shape[1]
    # subM = cluster_destinations.shape[0]
    K = cluster_destinations_old.shape[0]
    for k in range(K):
        subk_new = cluster_destinations_new[k]
        subk_old = cluster_destinations_old[k]
        if subk_new != subk_old:
            changed += 1
            for t in range(T):
                centers[subk_new, t] += n_kt_refined[k, t]
                centers[subk_old, t] -= n_kt_refined[k, t]
    return changed
    

cpdef _update_phi_centers(int[:,:] centers, double[:,:] phi_centers):
    # center_weight = centers.sum(axis=1)
    center_weight = np.sum(centers, axis=1)
    phi_centers = centers / np.c_[center_weight]


cpdef _infant_cluster(int[:,:] n_kt_refined, int subK):
    cdef:
        int K, T, subk, k

    K = n_kt_refined.shape[0]
    T = n_kt_refined.shape[1]
    # k_weight  = n_kt_refined.sum(axis=1)
    k_weight  = np.sum(n_kt_refined, axis=1)
    phi_refined = n_kt_refined / np.c_[k_weight]
    topK_topics = k_weight.argsort()[::-1]
    centers = np.empty((subK, T), dtype=np.intc)
    # center_weight = centers.sum(axis=1)
    center_weight = np.sum(centers, axis=1)
    phi_centers = centers / np.c_[center_weight]
    for subk in range(subK):
        centers[subk] = n_kt_refined[topK_topics[subk]]
    cluster_destinations_old = -1 * np.ones(K, dtype=np.intc)
    cluster_destinations_new = -1 * np.ones(K, dtype=np.intc)
    _assign_destination(phi_refined, phi_centers, cluster_destinations_old)
    _update_phi_centers(centers, phi_centers)
    _update_centers_first(centers, n_kt_refined, cluster_destinations_old)
    while True:
        _assign_destination(phi_refined, phi_centers, cluster_destinations_new)
        changed = _update_centers(centers, n_kt_refined,
                                  cluster_destinations_old,
                                  cluster_destinations_new)
        _update_phi_centers(centers, phi_centers)
        if changed <= 0:
            break
        for k in range(K):
            cluster_destinations_old[k] = cluster_destinations_new[k]
        print(changed)
    return centers
