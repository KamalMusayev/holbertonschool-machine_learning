#!/usr/bin/env python3
"""Function is starting"""


def matrix_transpose(matrix):
    """
    Inside of Function
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
