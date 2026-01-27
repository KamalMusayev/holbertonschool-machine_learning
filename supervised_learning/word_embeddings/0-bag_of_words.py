#!/usr/bin/env python3
"""Comment of Function"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Bag of Words"""
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)

    features = vectorizer.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features
    # Fill embeddings matrix
    for i, sent in enumerate(tokenized):
        for word in sent:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, features