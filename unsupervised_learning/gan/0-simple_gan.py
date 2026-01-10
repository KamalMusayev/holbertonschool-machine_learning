#!/usr/bin/env python3
"""Simple GAN Implementation"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class Simple_GAN(keras.Model):
    """Simple GAN class"""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.001):
        """
        Initialize Simple GAN

        Args:
            generator: generator network
            discriminator: discriminator network
            latent_generator: function to generate latent space samples
            real_examples: real data examples
            batch_size: batch size for training
            disc_iter: number of discriminator iterations per generator iteration
            learning_rate: learning rate for optimizers
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        # Compile generator and discriminator with optimizers
        self.generator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )

    def get_fake_sample(self, size=None, training=False):
        """Generate fake samples"""
        if not size:
            size = self.batch_size
        latent_sample = self.latent_generator(size)
        return self.generator(latent_sample, training=training)

    def get_real_sample(self, size=None):
        """Get real samples"""
        if not size:
            size = self.batch_size
        indices = tf.random.uniform(
            shape=(size,),
            minval=0,
            maxval=tf.shape(self.real_examples)[0],
            dtype=tf.int32
        )
        return tf.gather(self.real_examples, indices)

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
            discr_fake = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(discr_fake)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}