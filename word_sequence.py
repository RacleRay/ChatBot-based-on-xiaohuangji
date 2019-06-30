#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Raclerrr
# datetime:2019/1/21 17:49
# software:PyCharm
# modulename:word_sequence
import numpy as np


class WordSequence(object):

    PAD_TAG = '<pad>'  # 填充
    UNK_TAG = '<unk>'  # 未知
    START_TAG = '<s>'  # 开始
    END_TAG = '</s>'  # 结束

    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self):
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }  # {Word: id}
        self.fited = False  # 是否训练的标记

    def to_index(self, word):
        """word >>> id"""
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK

    def to_word(self, index):
        """id >>> word"""
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        for k, v in self.dict.items():
            if v == index:
                return k
        return WordSequence.UNK_TAG

    def size(self):
        assert self.fited, "WordSequence 尚未进行 fit 操作"
        return len(self.dict) + 1

    def __len__(self):
        return self.size()

    def fit(self, sentences, min_count=5, max_count=None, max_features=None):
        """对输入文本做词表统计
        args: sentences——输入文本；min_count——最小允许词频；
              max_count——最大允许词频；max_features——词表最多词汇量
        return：无
        """
        assert not self.fited, 'Word Sequence只能fit一次'
        count = {}

        # 全统计
        for sentence in sentences:
            arr = list(sentence)
            for a in arr:
                if a not in count:
                    count[a] = 0
                count[a] += 1

        # 词频筛选
        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}
        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}

        # 初始化
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }

        # 词表长度
        if isinstance(max_features, int):
            count = sorted(list(count.items()), key=lambda x: x[1])
            if len(count) > max_features:
                count = count[-int(max_features):]
            for w, _ in count:  # 以长度为id
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)

        self.fited = True

    def transform(self, sentence, max_len=None):
        """将句子转换为id
        args：sentence——输入句子；max_len——限制的最大句子长度
        return: ids of a sentence, np.array
        """
        assert self.fited, 'WordSequence 尚未进行 fit 操作'

        # 填充padding
        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)

        # 输出ids（可优化）
        for index, a in enumerate(sentence):
            if max_len is not None and index >= max_len:
                break
            r[index] = self.to_index(a)

        return np.array(r)

    def inverse_transform(self,
                          indices,
                          ignore_pad=False,
                          ignore_unk=False,
                          ignore_start=False,
                          igonre_end=False):
        """将一个序列的ids转换为句子
        args：indices——输入ids；ignore_pad——是否忽略PAD；
              ignore_unk——是否忽略UNK；ignore_start——是否忽略START；
              igonre_end——是否忽略END
        return: sentence——ids对应的词组成的句子, list
        """
        sentence = []
        for i in indices:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and igonre_end:
                continue
            sentence.append(word)
        return sentence


def test():
    ws = WordSequence()
    ws.fit([
        ['你', '好', '啊'],
        ['你', '好', '哦'],
    ])

    indice = ws.transform(['我', '们', '好'])
    print(indice)

    back = ws.inverse_transform(indice)
    print(back)


if __name__ == '__main__':
    test()