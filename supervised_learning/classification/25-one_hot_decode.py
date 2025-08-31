#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""


import numpy as np


def one_hot_decode(one_hot):
    """One-Hot Decode Function"""
    try:
        one_hot_decode = np.argmax(one_hot, axis=0)
        return one_hot_decode
    except Exception:
        return None
