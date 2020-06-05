#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ranking_questions.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import os
import pickle
import pandas as pd
import numpy as np

from util import load_embeddings, question_to_vec


def precompute_tag_specific_embedding():
    os.makedirs('embedding_of_differ_tags', exist_ok=True)

    embeddings, embeddings_dim = load_embeddings('stackoverflow_out_model.tsv')

    data_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')
    counts_by_tag = data_df.groupby('tag').count()['title']

    # 根据tag，分别储存question embedding
    for tag, count in counts_by_tag.items():
        tag_posts = data_df[data_df['tag'] == tag]
        tag_post_ids = tag_posts['post_id'].values

        tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
        for i, title in enumerate(tag_posts['title']):
            tag_vectors[i, :] = question_to_vec(title, embeddings, embeddings_dim)

        filename = os.path.join('embedding_of_differ_tags', os.path.normpath('%s.pkl' % tag))
        pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))


if __name__=='__main__':
    precompute_tag_specific_embedding()