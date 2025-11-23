#!/usr/bin/env python3
"""Advanced Linear Algebra"""


def determinant(matrix):
    """Determinant of Matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for x in range(len(matrix)):
        sub_matrix = [row[:x] + row[x + 1:] for row in matrix[1:]]
        det += ((-1) ** x) * matrix[0][x] * determinant(sub_matrix)

    return det


def minor(matrix):
    """Minor Matrix of Matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(sub_list):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]

    minor = []
    for a in range(len(matrix)):
        minor_ = []
        for b in range(len(matrix[0])):
            copy_matrix = [x[:] for x in matrix]
            del copy_matrix[a]
            for row in copy_matrix:
                del row[b]
            minor_.append(determinant(copy_matrix))
        minor.append(minor_)

    return minor


def cofactor(matrix):
    """Cofactor Matrix of Matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(sub_list):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]

    cofactor_matrix = []
    for a in range(len(matrix)):
        row = []
        for b in range(len(matrix[0])):
            copy_matrix = [x[:] for x in matrix]
            del copy_matrix[a]
            for r in copy_matrix:
                del r[b]
            c = (-1) ** (a + b)
            row.append(c * determinant(copy_matrix))
        cofactor_matrix.append(row)

    return cofactor_matrix


def adjugate(matrix):
    """Adjugate Matrix of Matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(sub_list):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]

    matrix = cofactor(matrix)
    adjugate = [[matrix[j][i] for j in range(len(matrix))]
                for i in range(len(matrix[0]))]

    return adjugate
