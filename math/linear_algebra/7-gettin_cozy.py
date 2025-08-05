#!/usr/bin/env python3
"""Function is starting"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Inside of Function
    """
    if axis == 0:
         return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1[:] + row2[:] for row1, row2 in zip(mat1, mat2)]
