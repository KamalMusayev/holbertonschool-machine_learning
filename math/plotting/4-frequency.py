#!/usr/bin/env python3
"""Plotting"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Inside of Function"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    bins = range(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.axis((None, None, 0, 30))
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xticks(bins)
    plt.show()
