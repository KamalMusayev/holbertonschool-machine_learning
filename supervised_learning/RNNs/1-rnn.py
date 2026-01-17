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
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t, m, h))
    Y = np.zeros((t, m, o))
    h_next = h_0.copy()

    for step in range(t):
        h_next, y = rnn_cell.forward(h_next, X[step])
        H[step] = h_next
        Y[step] = y

    return H, Y
