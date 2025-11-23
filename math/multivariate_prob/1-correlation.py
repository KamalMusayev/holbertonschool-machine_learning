#!/usr/bin/env python3
"""Multivariate Probability"""
import numpy as np


def correlation(C):
    """Correlation Matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1] or C.ndim != 2:
        raise ValueError("C must be a 2D square matrix")
    std = np.sqrt(np.diag(C))
    denom = np.outer(std, std)
    R = C / denom
    np.fill_diagonal(R, 1.0)

    return R
