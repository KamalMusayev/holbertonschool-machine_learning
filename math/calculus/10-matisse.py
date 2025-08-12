#!/usr/bin/env python3
"""Calculus"""


def poly_derivative(poly):
    """Inside of Function"""
    if type(poly) is not list:
        return None
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None
    if len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]

    poly_i = []
    for i in range(1, len(poly)):
        poly_i.append(poly[i] * i)

    return poly_i
