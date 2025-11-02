#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


class NST:
  """Class of NST"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize variables"""
        if not isinstance(style_image, np.ndarray):
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        
        if not isinstance(content_image, np.ndarray):
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
    
    @staticmethod
    def scale_image(image):
        """Static Method hat rescales an image
        such that its pixels values are
        between 0 and 1 and its largest side is 512 pixels"""
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        
        h, w, _ = image.shape
        
        if h > w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))
        
        image_tensor = tf.convert_to_tensor(image)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        
        scaled_image = tf.image.resize(
            image_tensor,
            size=(new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )
        
        scaled_image = scaled_image / 255.0
        scaled_image = tf.clip_by_value(scaled_image, 0.0, 1.0)
        
        return scaled_image
    
    def load_model(self):
        """Load Moodel"""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                idx = vgg.layers.index(layer)
                vgg.layers[idx] = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )
        
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]
        
        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
