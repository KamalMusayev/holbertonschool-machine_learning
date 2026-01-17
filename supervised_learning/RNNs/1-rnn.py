#!/usr/bin/env python3
"""
Module rnn
Performs forward propagation for a simple RNN over all time steps.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Parameters:
    rnn_cell (RNNCell): Instance of RNNCell used for forward propagation
    X (numpy.ndarray): Input data of shape (t, m, i)
    h_0 (numpy.ndarray): Initial hidden state of shape (m, h)

    Returns:
    H (numpy.ndarray): All hidden states of shape (t, m, h)
    Y (numpy.ndarray): All outputs of shape (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y = np.zeros((t, m, o))

    for step in range(t):
        x_t = X[step]
        h_prev = H[step]
        h_t, y_t = rnn_cell.forward(x_t, h_prev)
        H[step + 1] = h_t
        Y[step] = y_t

    return H, Y
