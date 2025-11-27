#!/usr/bin/env python3
import numpy as np


def pca(X, var=0.95):
    cov = np.dot(X.T, X) / X.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    total_var = np.sum(eigenvalues)
    cumulative_var = np.cumsum(eigenvalues) / total_var
    k = np.argmax(cumulative_var >= var) + 1
    W = eigenvectors[:, :k]
    return W
