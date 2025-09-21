#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """L2 Regularization Create Layer"""
    tf.random.set_seed(0)
    regularizer = tf.keras.regularizers.l2(lambtha)
    output_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer
    )(prev)
    return output_layer
