#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or type(k) is not int or k <= 0:
      return None, None
    
    pi = np.full((k, ), 1/k)

    C, clss = kmeans(X, k)
    m = C

    d = X.shape[1]
    S = np.array([np.eye(d) for _ in range(k)])

    return pi, m, S
