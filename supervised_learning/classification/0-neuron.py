#!/usr/bin/env python3
"""Classification"""


import numpy as np


class Neuron:
    """Neuron Class"""


    def __init__(self, nx, W, b, A):
        """Init Function"""
        if nx is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn()
        self.b = 0
        self.A = 0
