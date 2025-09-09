#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data

def create_mini_batches(X, Y, batch_size):
    """Mini Batches"""
    i = X.shape[0]
    perm = np.random.permutation(i)
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]
    mini_batch = []
    for first_ind in range(0, i, batch_size):
        last_ind = min(first_ind + batch_size, i)
        X_batch = X_shuffled[first_ind:last_ind]
        Y_batch = Y_shuffled[first_ind:last_ind]
        mini_batch.append((X_batch, Y_batch))
    return mini_batch
