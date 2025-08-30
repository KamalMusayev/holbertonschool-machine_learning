#!/usr/bin/env python3


"""Classification"""


import numpy as np


class DeepNeuralNetwork:
   def __init__(self, nx, layers):
        """Init Function"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev_nodes = nx

        for layer_idx, nodes in enumerate(layers, 1):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.weights["W" + str(layer_idx)] = np.random.randn(
                nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            self.weights["b" + str(layer_idx)] = np.zeros((nodes, 1))
            prev_nodes = nodes
