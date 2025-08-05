#!/usr/bin/env python3
"""Function is starting"""


def mat_mul(mat1, mat2):
    """
    Inside of Function
    """
    if len(mat1[0]) != len(mat2):
        return None

    new_mat = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            total = 0
            for k in range(len(mat2)):
                total += mat1[i][k] * mat2[k][j]
            row.append(total)
        new_mat.append(row)

    return new_mat
