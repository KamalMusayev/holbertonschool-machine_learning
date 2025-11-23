#!/usr/bin/env python3
"""Multivariate Probability"""
import numpy as np


class MultiNormal:
    """Multivariate Normal distribution"""
    def __init__(self, data):
        """Initialize variables"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_centered = data - self.mean

        self.cov = (data_centered @ data_centered.T) / (n - 1)
        self.d = d

    def pdf(self, x):
        """PDF Function"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        x = np.array(x)
        diff = x - self.mean
        det = np.linalg.det(self.cov)
        if det == 0:
            return 0
        inv = np.linalg.inv(self.cov)

        exponent = -0.5 * np.dot(diff.T, np.dot(inv, diff))

        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** self.d) * det)

        return norm_const * np.exp(exponent)
