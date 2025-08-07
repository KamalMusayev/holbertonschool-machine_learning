#!/usr/bin/env python3
"""Function is starting"""


import numpy as np


def add_matrices(mat1, mat2):
    """
    Inside of Function
    """
    mat3 = np.array(mat1)
    mat4 = np.array(mat2)
    if mat3.shape != mat4.shape:
        return None

    return mat3 + mat4
