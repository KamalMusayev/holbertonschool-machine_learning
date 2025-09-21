#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Dropout Forward Propagation"""
    cache = {}
    A_prev = X
    for l in range(1, L):
        W = weights["W" + str(l)]
        b = weights["b" + str(l)]
        Z = np.dot(W, A_prev) + b
        A = np.tanh(Z)

        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
        A = A * D
        A = A / keep_prob

        cache["A" + str(l)] = A
        cache["D" + str(l)] = D
        A_prev = A

        WL = weights["W" + str(L)]
        bL = weights["b" + str(L)]
        ZL = np.dot(WL, A_prev) + bL
        AL = np.exp(ZL) / np.sum(np.exp(ZL), axis=0, keepdims=True)
        cache["A" + str(L)] = AL

        return cache
