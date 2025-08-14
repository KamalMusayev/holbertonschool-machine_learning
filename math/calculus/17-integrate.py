#!/usr/bin/python3
"""Calculus"""


def poly_integral(poly, C=0):
    """Inside of Function"""
    if type(poly) is not list:
        return None
    if not isinstance(C, int) or isinstance(C, float):
        return None
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None
    if len(poly) == 0:
        return None
    poly_integ = []
    poly_integ.append(C)
    for i in range(len(poly)):
        poly_integ.append(poly[i]/(i + 1))

    poly_integ = [int(c) if isinstance(c, float) and
                            c.is_integer() else c for c in poly_integ]
    while len(poly_integ) > 1 and poly_integ[-1] == 0:
        poly_integ.pop()
    return poly_integ
