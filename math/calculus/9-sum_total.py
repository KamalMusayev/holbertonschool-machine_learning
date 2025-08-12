#!/usr/bin/env python3
"""Calculus"""


def summation_i_squared(n):
    """Inside of Function"""
    if type(n) is not int or n <= 0:
        return None
    else:
        return int((n * (n+1) * (2*n+1))/6)
