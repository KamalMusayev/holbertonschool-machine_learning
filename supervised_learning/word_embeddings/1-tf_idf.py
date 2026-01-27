#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np
import re

def tf_idf(sentences, vocab=None):
    """TF-IDF"""
    tokenized_sentences = []
    for sentence in sentences:
        tokens = re.findall(r"\b[\w']+\b", sentence.lower())
        tokenized_sentences.append(tokens)

    if vocab is None:
        all_words = set()
        for tokens in tokenized_sentences:
            all_words.update(tokens)
        features = sorted(all_words)
    else:
        features = vocab

    word_to_index = {word: idx for idx, word in enumerate(features)}

    num_sentences = len(sentences)
    num_features = len(features)

    tf_matrix = np.zeros((num_sentences, num_features))

    for i, tokens in enumerate(tokenized_sentences):
        for token in tokens:
            if token in word_to_index:
                tf_matrix[i, word_to_index[token]] += 1

        if len(tokens) > 0:
            tf_matrix[i] = tf_matrix[i] / len(tokens)

    idf_vector = np.zeros(num_features)

    for j, word in enumerate(features):
        doc_count = 0
        for tokens in tokenized_sentences:
            if word in tokens:
                doc_count += 1

        if doc_count > 0:
            idf_vector[j] = np.log(num_sentences / doc_count)
        else:
            idf_vector[j] = 0

    embeddings = tf_matrix * idf_vector

    return embeddings, np.array(features)