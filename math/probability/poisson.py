#!/usr/bin/env python3
"""Probability"""
import math



class Poisson:
    """Poisson Class"""
    def __init__(self, data=None, lambtha=1.):
        """Initiailize Function"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """Probability Mass Function)"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e_term = math.exp(-self.lambtha)
        numerator = (self.lambtha ** k) * e_term
        denominator = math.factorial(k)
