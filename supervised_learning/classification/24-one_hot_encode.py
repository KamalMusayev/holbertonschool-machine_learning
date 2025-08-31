#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""


import numpy as np


def one_hot_encode(Y, classes):
    """One-Hot Encode Function"""
    m = Y.shape[1]
    one_hot = np.zeros(classes, m)
    one_hot[Y, np.arange(m)] = 1
    return one_hot
