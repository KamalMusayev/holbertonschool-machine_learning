#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the BIC"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None, None, None
    if kmax is not None and kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    expectation_maximization = __import__('8-EM').expectation_maximization

    k_range = kmax - kmin + 1
    log_likelihoods = np.zeros(k_range)
    bic_values = np.zeros(k_range)
    results = []

    best_bic = None
    best_k = None
    best_result = None

    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)

        if pi is None:
            return None, None, None, None

        log_likelihoods[i] = log_likelihood

        p = k * d + k * d * (d + 1) / 2 + k - 1

        bic = p * np.log(n) - 2 * log_likelihood
        bic_values[i] = bic

        results.append((pi, m, S))

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bic_values
