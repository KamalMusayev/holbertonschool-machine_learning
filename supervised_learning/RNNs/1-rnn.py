#!/usr/bin/env python3
"""
Module rnn
Performs forward propagation for a simple RNN over all time steps.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t, m, h))
    Y = np.zeros((t, m, o))
    h_next = h_0.copy()

    for step in range(t):
        H[step] = h_next
        h_next, y = rnn_cell.forward(h_next, X[step])
        Y[step] = y

    return H, Y
