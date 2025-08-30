#!/usr/bin/env python3


"""Classification"""


import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """Init Function"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(self.__L):
            nodes = layers[l]
            prev_nodes = nx if l == 0 else layers[l - 1]
            self.__weights["W" + str(l + 1)] = np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            self.__weights["b" + str(l + 1)] = np.zeros((nodes, 1))
