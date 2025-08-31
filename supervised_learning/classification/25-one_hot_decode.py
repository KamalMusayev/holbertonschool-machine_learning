#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""


import numpy as np


def one_hot_decode(one_hot):
    """One-Hot Decode Function"""
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
