#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """L2 Regularization Create Layer"""
    # He normal başlatma üsulunu təyin edirik
    initializer = tf.keras.initializers.HeNormal()
    
    # L2 requlyarizatorunu təyin edirik
    regularizer = tf.keras.regularizers.l2(lambtha)
    
    # Layer-i initializer və regularizer ilə yaradırıq
    output_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,  # <-- Əlavə edilməli olan sətir
        kernel_regularizer=regularizer
    )(prev)
    
    return output_layer
