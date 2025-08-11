#!/usr/bin/env python3
"""Plotting"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Inside of Function"""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y)
    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.yscale("log")
    plt.axis((0, 28650, None, None))
    plt.show()
