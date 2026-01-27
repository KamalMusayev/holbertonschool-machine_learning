#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Bag of Words"""
    tokenized = []
    for sentence in sentences:
        tokens = sentence.lower().split()
        tokenized.append(tokens)

    # Create vocabulary
    if vocab is None:
        features = sorted(set(word for sent in tokenized for word in sent))
    else:
        features = vocab

    # Initialize embeddings matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Create word to index mapping for efficiency
    word_to_idx = {word: idx for idx, word in enumerate(features)}

    # Fill embeddings matrix
    for i, sent in enumerate(tokenized):
        for word in sent:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, features