#!/usr/bin/env python3
"""Probability"""


class Binomial:
    """Binomial Class"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initiailize Function"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1 - (var / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est
            self.n = n_est
            self.p = p_est
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
