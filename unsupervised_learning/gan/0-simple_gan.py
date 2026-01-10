#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


def train_step(self, useless_argument):
    """Train Step"""
    for _ in range(self.disc_iter):

        real_sample = self.get_real_sample()
        fake_sample = self.get_fake_sample(training=True)

        with tf.GradientTape() as tape:
            discr_real = self.discriminator(real_sample, training=True)
            discr_fake = self.discriminator(fake_sample, training=True)
            discr_loss = self.discriminator.loss(discr_real, discr_fake)

        grads = tape.gradient(discr_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )

    with tf.GradientTape() as tape:
        fake_sample = self.get_fake_sample(training=True)
        discr_fake  = self.discriminator(fake_sample, training=False)
        gen_loss    = self.generator.loss(discr_fake)

    grads = tape.gradient(gen_loss, self.generator.trainable_variables)
    self.generator.optimizer.apply_gradients(
        zip(grads, self.generator.trainable_variables)
    )

    return {"discr_loss": discr_loss, "gen_loss": gen_loss}