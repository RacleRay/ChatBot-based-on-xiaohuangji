#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dialogue_manager.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from util import *


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)

        best_thread = pairwise_distances_argmin(
            X=question_vec.reshape(1, self.embeddings_dim),
            Y=thread_embeddings,
            metric='cosine'
        )
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

        # Chit-chat part
        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        self.chitchatBot = ChatBot("ChatBot")
        trainer = ListTrainer(self.chitchatBot)
        conversation = [
            "Hello",
            "Hi there!",
            "How are you doing?",
            "I'm doing great.",
            "That is good to hear",
            "Thank you.",
            "You're welcome."
        ]
        trainer.train(conversation)

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""
        question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([question])
        intent = self.intent_recognizer.predict(features)[0]

        if intent == 'dialogue':
            response = self.chitchatBot.get_response(question)
            return response
        else:
            tag = self.tag_classifier.predict(features)[0]
            thread_id = self.thread_ranker.get_best_thread(question, tag)
            return self.ANSWER_TEMPLATE % (tag, thread_id[0])


if __name__=='__main__':
    bot = DialogueManager(RESOURCE_PATH)
    question = 'how to use python list'
    print(bot.generate_answer(question))
