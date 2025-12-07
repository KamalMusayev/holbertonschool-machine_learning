#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the BIC"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    log_likelihoods = np.zeros(kmax - kmin + 1)
    BIC_values = np.zeros(kmax - kmin + 1)

    best_k = None
    best_bic = float('inf')
    best_result = None

    for k in range(kmin, kmax + 1):
        pi, m, S, g, l = expectation_maximization(X, k, iterations=iterations, tol=tol, verbose=verbose)

        if pi is None:
            continue

        p = k * (d + (d * (d + 1)) // 2 + 1)

        BIC_value = p * np.log(n) - 2 * l

        log_likelihoods[k - kmin] = l
        BIC_values[k - kmin] = BIC_value

        if BIC_value < best_bic:
            best_bic = BIC_value
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, BIC_values
