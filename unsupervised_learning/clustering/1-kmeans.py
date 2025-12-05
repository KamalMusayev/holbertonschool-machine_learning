#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if type(X) is not np.ndarray or type(k) is not int or k <= 0:
        return None

    if len(X.shape) != 2:
        return None

    n, d = X.shape
    if n == 0:
        return None
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    centroids = np.random.uniform(mins, maxs, size=(k, d))
    return centroids

def kmeans(X, k, iterations=1000):
    """K-means"""
    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    clss = np.zeros(n, dtype=int)

    for it in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)
        new_clss = np.argmin(distances, axis=1)

        if np.array_equal(new_clss, clss):
            break

        clss = new_clss.copy()

        for i in range(k):
            points_in_cluster = X[clss == i]
            if len(points_in_cluster) == 0:
                mins = X.min(axis=0)
                maxs = X.max(axis=0)
                C[i] = np.random.uniform(mins, maxs)
            else:
                C[i] = points_in_cluster.mean(axis=0)

    return C, clss
