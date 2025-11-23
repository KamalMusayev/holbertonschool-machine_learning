#!/usr/bin/env python3
"""Advanced Linear Algebra"""
import numpy as np


def definiteness(matrix):
    """Definiteness of Matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    n, m = matrix.shape
    if matrix.ndim != 2 or n != m:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except Exception:
        return None

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
