#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Dropout Forward Propagation"""
    np.random.seed(0)
    cache = {}
    cache['A0'] = X
    A = X
    # Loop for hidden layers (1 to L-1)
    for l in range(1, L):
        A_prev = A
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        # All hidden layers use tanh activation
        A = np.tanh(Z)
        # Apply dropout to the hidden layer's activation
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A = A * D
        A = A / keep_prob
        cache['D' + str(l)] = D
        cache['A' + str(l)] = A

    # Calculation for the output layer (L)
    A_prev = A
    W = weights['W' + str(L)]
    b = weights['b' + str(L)]
    Z = np.dot(W, A_prev) + b
    # The output layer uses softmax activation
    t = np.exp(Z)
    A = t / np.sum(t, axis=0, keepdims=True)
    cache['A' + str(L)] = A

    return cache
