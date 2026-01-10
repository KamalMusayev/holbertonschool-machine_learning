#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class Simple_GAN(keras.Model):
    """Class"""


    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.001):
        """
        __init__
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.gen_loss_fn = lambda pred: tf.keras.losses.MSE(tf.ones_like(pred), pred)
        self.disc_loss_fn = lambda real, fake: tf.keras.losses.MSE(tf.ones_like(real), real) + \
                                               tf.keras.losses.MSE(-tf.ones_like(fake), fake)

        self.generator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    def get_fake_sample(self, size=None, training=False):
        """
        get_fake_sample
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        get_real_sample
        """
        if not size:
            size = self.batch_size
        indices = tf.random.uniform(shape=(size,), minval=0,
                                    maxval=tf.shape(self.real_examples)[0],
                                    dtype=tf.int32)
        return tf.gather(self.real_examples, indices)

    def train_step(self, useless_argument):
        """
        train_step
        """
        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample(training=True)
            with tf.GradientTape() as tape:
                discr_real = self.discriminator(real_sample, training=True)
                discr_fake = self.discriminator(fake_sample, training=True)
                discr_loss = self.disc_loss_fn(discr_real, discr_fake)

            grads = tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            discr_fake = self.discriminator(fake_sample, training=False)
            gen_loss = self.gen_loss_fn(discr_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}