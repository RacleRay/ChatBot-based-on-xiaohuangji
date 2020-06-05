#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_intent_classifier.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from util import text_prepare


def tfidf_features(X_train, X_test, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""
    tfidf_vectorizer = TfidfVectorizer(min_df=5,
                                       max_df=0.9,
                                       token_pattern='(\S+)',
                                       ngram_range=(1, 2))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)

    fileObject = open(vectorizer_path, 'wb')
    pickle.dump(tfidf_vectorizer, fileObject)
    fileObject.close()

    return X_train, X_test, tfidf_vectorizer


def read_and_preprocess(data_path, col_name, sample_size=200000):
    df = pd.read_csv(data_path, sep='\t').sample(sample_size, random_state=0)
    df[col_name] = [text_prepare(x) for x in df[col_name]]
    return df


def gen_intent_train_test(pos_file, neg_file, output_path):
    "Generate train and test file, transfrom to tfidf_features"
    pos = read_and_preprocess(pos_file, 'title')
    neg = read_and_preprocess(neg_file, 'text')

    X = np.concatenate([neg['text'].values, pos['title'].values])
    y = ['dialogue'] * neg.shape[0] + ['stackoverflow'] * pos.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=0)
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_features(
        X_train, X_test, output_path)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer


def classify_intent(X_train, y_train, X_test, y_test, output_path):
    intent_cla = LogisticRegression(penalty='l2', C=10, random_state=0)
    intent_cla.fit(X_train, y_train)

    pickle.dump(intent_cla, open(output_path, 'wb'))

    y_test_pred = intent_cla.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Intent test accuracy = {}'.format(test_accuracy))


def gen_language_train_test(pos_file, tfidf_vectorizer):
    pos_df = read_and_preprocess(pos_file, 'title')

    X = pos_df['title'].values
    y = pos_df['tag'].values

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)
    X_train_tfidf = tfidf_vectorizer.transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test


def classify_tag(X_train, X_test, y_train, y_test, output_path):
    tag_classifier = OneVsRestClassifier(
        LogisticRegression(penalty='l2', C=5, random_state=0))
    tag_classifier.fit(X_train, y_train)

    y_test_pred = tag_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Tag test accuracy = {}'.format(test_accuracy))

    pickle.dump(tag_classifier, open(output_path, 'wb'))


if __name__ == '__main__':
    pos_file = 'data/tagged_posts.tsv'
    neg_file = 'data/dialogues.tsv'
    output_path_tfidf = 'model/tfidf_vectorizer.pkl'
    output_path_intent = 'model/intent_recognizer.pkl'
    output_path_tag = 'model/tag_classifier.pkl'

    X_train_intent, X_test_intent, y_train_intent, y_test_intent, tfidf_vectorizer = gen_intent_train_test(
        pos_file, neg_file, output_path_tfidf)
    classify_intent(X_train_intent, y_train_intent, X_test_intent,
                    y_test_intent, output_path_intent)

    X_train_tag, X_test_tag, y_train_tag, y_test_tag = gen_language_train_test(
        pos_file, tfidf_vectorizer)
    classify_tag(X_train_tag, X_test_tag, y_train_tag, y_test_tag, output_path_tag)
