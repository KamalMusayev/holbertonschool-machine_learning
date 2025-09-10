#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Learning Rate Decay"""
    a = tf.keras.optimizers.schedules.InverseTimeDecay(alpha, decay_rate, decay_step)
    return a
