#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """TF-IDF"""
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

    features = np.array(vocab)

    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    s = len(sentences)
    f = len(vocab)

    tf_matrix = np.zeros((s, f), dtype=float)

    for i, words in enumerate(tokenized_sentences):
        word_count = {}
        for word in words:
            if word in word_to_idx:
                word_count[word] = word_count.get(word, 0) + 1

        total_words = len(words) if words else 1
        for word, count in word_count.items():
            idx = word_to_idx[word]
            tf_matrix[i, idx] = count / total_words

    idf_vector = np.zeros(f, dtype=float)

    for j, word in enumerate(vocab):
        doc_count = 0
        for words in tokenized_sentences:
            if word in words:
                doc_count += 1

        if doc_count > 0:
            idf_vector[j] = np.log((s + 1) / (doc_count + 1)) + 1
        else:
            idf_vector[j] = 1

    tfidf_matrix = tf_matrix * idf_vector

    for i in range(s):
        norm = np.linalg.norm(tfidf_matrix[i])
        if norm > 0:
            tfidf_matrix[i] = tfidf_matrix[i] / norm

    return tfidf_matrix, features
