#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Convolve Grayscale Valid"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = (kh - 1) // 2
    pw = (kw - 1) // 2
    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant',
                           constant_values=0)
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            image_patch = images_padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_patch * kernel, axis=(1, 2))
    return output
