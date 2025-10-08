#!/usr/bin/env python3
"""Comment of Function"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Inception Network"""
    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(F1, (1, 1), padding='same',
                            activation='relu')(A_prev)

    conv3 = K.layers.Conv2D(F3R, (1, 1), padding='same',
                            activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            activation='relu')(conv3)

    conv5 = K.layers.Conv2D(F5R, (1, 1), padding='same',
                            activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(F5, (5, 5), padding='same',
                            activation='relu')(conv5)

    pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    pool = K.layers.Conv2D(FPP, (1, 1), padding='same',
                           activation='relu')(pool)

    output = K.layers.concatenate([conv1, conv3, conv5, pool])

    return output
