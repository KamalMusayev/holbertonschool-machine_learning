#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build Model Function"""
    model = Sequential()
    model.add(Dense(layers[0],
                    activation = activations[0],
                    kernel_regularizer = l2(lambtha),
                    input_shape=(nx,)))
    model.add(Dropout(1 - keep_prob))
    for i in range(1, len(layers)):
        model.add(Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=l2(lambtha)
        ))
        if i < len(layers) - 1:
            model.add(Dropout(1 - keep_prob))
    return model
