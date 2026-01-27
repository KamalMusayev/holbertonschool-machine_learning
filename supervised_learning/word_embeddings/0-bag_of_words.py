#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Bag of Words"""
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]

    # Determine features/vocabulary
    if vocab is None:
        # Extract all unique words and sort them
        all_words = set()
        for tokens in tokenized_sentences:
            all_words.update(tokens)
        features = sorted(all_words)
    else:
        features = vocab

    # Create word-to-index mapping for efficient lookup
    word_to_index = {word: idx for idx, word in enumerate(features)}

    # Initialize embedding matrix
    num_sentences = len(sentences)
    num_features = len(features)
    embeddings = np.zeros((num_sentences, num_features), dtype=int)

    # Fill the embedding matrix
    for i, tokens in enumerate(tokenized_sentences):
        for token in tokens:
            if token in word_to_index:
                embeddings[i, word_to_index[token]] += 1

    return embeddings, features