#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """L2 Regularization Create Layer"""
    kernel_regularizer = tf.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_regularizer=kernel_regularizer)
    output_layer = layer(prev)
    return output_layer
