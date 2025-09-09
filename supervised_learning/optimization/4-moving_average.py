#!/usr/bin/env python3

"""Comment of Function"""

import numpy as np


def moving_average(data, beta):
    """Moving Average"""
    mov_avg = []
    for i in range(len(data) - beta + 1):
        avg = np.mean(data[i: i + beta])
        mov_avg.append(avg)
    return np.array(mov_avg)
