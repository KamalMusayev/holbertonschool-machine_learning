#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf

def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update Variables Momentum"""
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
