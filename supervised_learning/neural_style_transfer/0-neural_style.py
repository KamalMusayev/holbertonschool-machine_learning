#!/usr/bin/env python3
"""Neural Style Transfer"""

import tensorflow as tf
import numpy as np

class NST:
    """Class for Neural Style Transfer"""
    
    # Public class attributes
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """NST class constructor"""
        
        # Type checks for style_image
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        # Type checks for content_image
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        # Type checks for alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        # Type checks for beta
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        
        # Preprocess and set instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
    
    @staticmethod
    def scale_image(image):
        """Rescales an image to have pixel values between 0 and 1
        and largest side equal to 512 pixels"""
        
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Compute scaling factor
        h, w, _ = image.shape
        if h >= w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))
        
        # Resize using bicubic interpolation
        image_resized = tf.image.resize(image, (new_h, new_w), method='bicubic')
        
        # Rescale pixel values to [0, 1]
        image_scaled = image_resized / 255.0
        
        # Add batch dimension
        image_scaled = tf.expand_dims(image_scaled, axis=0)
        
        return image_scaled
