#!/usr/bin/env python3
"""Comment of Function"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
  """ResNet-50"""
  input_layer = K.Input(shape=(224,224,3))
  he_normal = K.initializers.HeNormal(seed=0)
  X = K.layers.Conv2D(64, (7,7), padding='same',
                      activation='relu')(input_layer)
  X = K.layers.BatchNormalization()(X)
  X = K.layers.ReLU()(X)
  X = K.layers.MaxPooling2D((3,3), strides=2 padding='same')(X)

  X = projection_block(X, filters=(64, 64, 256), s=1)
  X = identity_block(X, filters=(64, 64, 256))
  X = identity_block(X, filters=(64, 64, 256))

  X = projection_block(X, filters=(128, 128, 512), s=2)
  X = identity_block(X, filters=(128, 128, 512))
  X = identity_block(X, filters=(128, 128, 512))
  X = identity_block(X, filters=(128, 128, 512))

  X = projection_block(X, filters=(256, 256, 1024), s=2)
  for _ in range(5):
      X = identity_block(X, filters=(256, 256, 1024))
  
  X = projection_block(X, filters=(512, 512, 2048), s=2)
  X = identity_block(X, filters=(512, 512, 2048))
  X = identity_block(X, filters=(512, 512, 2048))

  X = layers.GlobalAveragePooling2D()(X)
  output_layer = layers.Dense(1000, activation='softmax',
                              kernel_initializer=he_normal)(X)

  model = Model(inputs=input_layer, outputs=output_layer)
  return model
