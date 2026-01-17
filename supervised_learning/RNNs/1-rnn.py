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
    h_next = h_0
    H = np.zeros((t, m, h))
    o = rnn_cell.Wy.shape[1]
    Y = np.zeros((t, m, o))

    for step in range(t):
        h_next, y = rnn_cell.forward(h_next, X[step])
        H[step] = h_next
        Y[step] = y

    return H, Y
