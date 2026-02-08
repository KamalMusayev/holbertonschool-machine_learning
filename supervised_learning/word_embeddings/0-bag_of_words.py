#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Bag of Words"""
    tokenized_sentences = []
    for sentence in sentences:
        cleaned = sentence.lower()
        for char in "!.,?;:\"":
            cleaned = cleaned.replace(char, " ")
        words = cleaned.split()
        processed_words = []
        for word in words:
            if word.endswith("'s"):
                word = word[:-2]
            elif word.endswith("'"):
                word = word[:-1]
            if word:
                processed_words.append(word)
        tokenized_sentences.append(processed_words)

    if vocab is None:
        vocab_set = set()
        for words in tokenized_sentences:
            vocab_set.update(words)
        vocab = sorted(list(vocab_set))
    else:
        vocab = list(vocab)

    features = vocab

    word_to_idx = {word: idx for idx, word in enumerate(features)}

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, words in enumerate(tokenized_sentences):
        for word in words:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, np.array(features)
