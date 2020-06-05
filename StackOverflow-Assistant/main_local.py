#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main_local.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

from dialogue_manager import DialogueManager
from util import RESOURCE_PATH

# NOTE: 本地执行的Bot

def is_unicode(text):
    return len(text) == len(text.encode())


def main():
    paths = RESOURCE_PATH

    bot = DialogueManager(paths)

    print("Ready to talk!")
    while True:
        question = input('>>>:')
        if is_unicode(question):
            answer = bot.generate_answer(question)
        else:
            answer = "Hmm, you are sending some weird characters to me..."
        print('Bot:', answer)


if __name__ == "__main__":
    main()