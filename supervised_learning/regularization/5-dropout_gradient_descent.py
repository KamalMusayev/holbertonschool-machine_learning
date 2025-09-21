#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def dropout_gradient_descent(Y, weights,
                             cache, alpha,
                             keep_prob, L):
    """Dropout Gradient Descent"""
    m = Y.shape[1]
    AL = cache["A" + str(L)]
    dZ = AL - Y
    for i in range(L, 0, -1):
        A_prev = cache["A{}".format(i - 1)]
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    weights["W{}".format(i)] -= alpha * dW
    weights["b{}".format(i)] -= alpha * db
        if i > 1:
            W = weights["W{}".format(i)]
            dA_prev = np.dot(W.T, dZ)
            D_prev = cache["D{}".format(i - 1)]
            dA_prev = dA_prev * D_prev / keep_prob
            A_prev = cache["A{}".format(i - 1)]
            dZ = dA_prev * (1 - A_prev ** 2)
