#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """L2 Regularization Cost"""
    l2_loss = tf.add_n(model.losses)
    total_costs = [cost + loss for loss in l2_loss]
    return total_costs
