#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def moving_average(data, beta):
    """Moving Average"""
    weights = np.ones(beta) / beta
    moving_avg = np.convolve(data, weights, mode='valid')
    return moving_avg
