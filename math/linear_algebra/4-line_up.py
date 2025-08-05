#!/usr/bin/env python3
"""Function is starting"""


def add_arrays(arr1, arr2):
    """
    Inside of Function
    """
    if len(arr1) != len(arr2):
        return None

    new_mat = []
    for i in range(len(arr1)):
        new_mat.append(arr1[i] + arr2[i])

    return new_mat
