#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Raclerrr
# datetime:2019/1/28 23:57
# software:PyCharm
# modulename:fake_data
import random
import numpy as np
from word_sequence import WordSequence

def generate(max_len=10, size=1000, same_len=False, seed=0):
    """
    生成测试方法用的数据
    :param max_len: seq最大长度
    :param size: seq数量
    :param same_len: x_data中的seq与y_data中的seq是否等长
    :param seed: 随机种子
    :return: x_data, y_data: list，伪问答对序列
             ws_input, ws_target：WordSequence对象，词表保存在self.dict中
    """
    dictionary = {
        'a': '1',
        'b': '2',
        'c': '3',
        'd': '4',
        'aa': '1',
        'bb': '2',
        'cc': '3',
        'dd': '4',
        'aaa': '1',
    }

    if seed is not None:
        random.seed(seed)

    # 输入词表
    input_list = sorted(list(dictionary.keys()))

    x_data = [] # seq
    y_data = [] # 标签

    # 生成伪问答对
    for i in range(size):
        a_len = int(random.random() * max_len) + 1
        x = []
        y = []
        for _ in range(a_len):
            word = input_list[int(random.random() * len(input_list))]
            x.append(word)
            y.append(dictionary[word])
            if not same_len:
                if y[-1] == '2':
                    y.append('2')
                elif y[-1] == '3':
                    y.append('3')
                    y.append('4')
        x_data.append(x)
        y_data.append(y)

    ws_input = WordSequence()
    ws_input.fit(x_data)

    ws_target = WordSequence()
    ws_target.fit(y_data)
    return x_data, y_data, ws_input, ws_target


def test():
    x_data, y_data, ws_input, ws_target = generate()
    print(len(x_data))
    assert len(x_data) == 1000
    print(len(y_data))
    assert len(y_data) == 1000
    print(np.max([len(x) for x in x_data]))
    assert np.max([len(x) for x in x_data]) == 10
    print(len(ws_input))
    print(ws_input.dict)
    assert len(ws_input) == 14
    print(len(ws_target))

if __name__ == '__main__':
    test()


