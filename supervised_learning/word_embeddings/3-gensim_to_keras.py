#!/usr/bin/env python3
"""Comment of Function"""
import tensorflow.keras as K

def gensim_to_keras(model):
    """Convert gensim word2vec to keras Embedding"""
    weights = model.wv.vectors

    embedding_layer = K.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding_layer