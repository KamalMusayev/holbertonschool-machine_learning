#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def f1_score(confusion):
    """F1 Score"""
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision
    f1 = ((2 * (precision * sensitivity)) /
          (precision + sensitivity))
    return f1
