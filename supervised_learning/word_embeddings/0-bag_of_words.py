#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Bag of Words"""
    tokenized = []

    for sentence in sentences:
        # only keep alphabetic words
        tokens = re.findall(r"[a-z]+", sentence.lower())
        tokenized.append(tokens)

    if vocab is None:
        features = sorted(set(word for sent in tokenized for word in sent))
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sent in enumerate(tokenized):
        for j, word in enumerate(features):
            embeddings[i, j] = sent.count(word)

    return embeddings, features
