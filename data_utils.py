#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Raclerrr
# datetime:2019/1/25 16:46
# software:PyCharm
# modulename:data_utils
import random
import numpy as np
from tensorflow.python.client import device_lib
from word_sequence import WordSequence

VOCAB_SIZE_THRESHOLD_CPU = 16000
# 由于gpu显存较小，设置阈值。大于该阈值使用cpu

def _get_available_gpus():
    """获取当前GPU信息"""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def _get_embed_device(vocab_size):
    """根据输入输出的字典大小来选择，是在CPU上embedding还是在GPU上进行embedding"""
    gpus = _get_available_gpus()
    if not gpus or vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    return "/gpu:0"


def transform_sentence(sentence, ws, max_len=None, add_end=False):
    """ 单独的句子转换为数组
    args: sentence——输入句子；
          ws——WordSequence对象；
          max_len——最大长度；
          add_end——加入结束符
    return: encoded——句子转换为的ids；
            encoded_len——句子长度，最大为max_len
    """
    encoded = ws.transform(
        sentence,
        max_len=max_len if max_len is not None else len(sentence))
    encoded_len = len(sentence) + (1 if add_end else 0)
    if encoded_len > len(encoded):
        encoded_len = len(encoded)
    return encoded, encoded_len


def batch_flow(data, ws, batch_size, raw=False, add_end=True):
    """
    从数据中随机去生成batch_size的数据，然后转换后输出
    :param data: list or tuple，包含问答对数据[x_data, y_data]，x_data：list，内多个字符串
                 y_data：list，内多个字符串
    :param ws: list or tuple, 包含[ws_input, ws_target], 为WordSequence对象
    :param batch_size: 批次容量
    :param raw: 是否输出原字符串
    :param add_end: 加入结束标记
    :return: batches，生成器生成，array数组
        raw = False:
        next(generator) == q_i_encoded, q_i_len, a_i_encoded, a_i_len
        raw = True:
        next(generator) == q_i_encoded, q_i_len, q_i, a_i_encoded, a_i_len, a_i
    """
    all_data = list(zip(*data))

    if isinstance(ws, (list, tuple)):
        assert len(ws) == len(data), \
            'ws的长度必须等于data的长度 if ws 是一个list or tuple'

    if isinstance(add_end, bool):
        add_end = [add_end] * len(data)
    else:
        assert(isinstance(add_end, (list, tuple))), \
            'add_end不是boolean，就应该是一个list(tuple) of boolean'
        assert len(add_end) == len(data), \
            '如果add_end是list(tuple)，那么add_end的长度应该和输入数据的长度一致'

    mul = 2
    if raw:
        mul = 3

    while True:
        data_batch = random.sample(all_data, batch_size)
        batches = [[] for i in range(len(data) * mul)]
        max_lens = []
        for j in range(len(data)):
            max_len = max([
                len(x[j]) if hasattr(x[j], '__len__') else 0
                for x in data_batch
            ])
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws, (list, tuple)):
                    w = ws[j]
                else:
                    w = ws
                # 添加结束标记（结尾）
                line = d[j]
                if add_end[j] and isinstance(line, (tuple, list)):
                    line = list(line) + [WordSequence.END_TAG]
                if w is not None:
                    x, xl = transform_sentence(line, w, max_lens[j], add_end[j])
                    batches[j * mul].append(x)
                    batches[j * mul + 1].append(xl)
                else:  # w为空，无法编码
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)

                if raw: # 加入原seq
                    batches[j * mul + 2].append(line)
            batches = [np.asarray(x) for x in batches]
            yield batches


def batch_flow_bucket(data, ws, batch_size, raw=False, add_end=True,
                      n_bucket=5, bucket_ind=1, debug=False):
    """
    根据seq长度对数据进行分组, 按组生成batch
    :param data: list or tuple，包含问答对数据[x_data, y_data]，x_data：list，内多个字符串
                 y_data：list，内多个字符串
    :param ws: list or tuple, 包含[ws_input, ws_target], 为WordSequence对象
    :param batch_size: 批次容量
    :param raw: 是否输出原字符串
    :param add_end: 加入结束标记
    :param n_bucket: 数据分成了多少个bucket
    :param bucket_ind: 是指哪一个维度的输入作为bucket的依据
    :param debug: 调试模式
    :return: batches，生成器生成，array数组
    """
    all_data = list(zip(*data))
    # 指定维度的seq所有长度的集合
    lengths = sorted(list(set([len(x[bucket_ind]) for x in all_data])))
    if n_bucket > len(lengths):
        n_bucket = len(lengths)

    # 长度划分分界线
    splits = np.array(lengths)[
        (np.linspace(0, 1, 5, endpoint=False) * len(lengths)).astype(int)
    ].tolist()
    splits += [np.inf]

    if debug:
        print("splits: ", splits)

    ind_data = {}
    for x in all_data:
        l = len(x[bucket_ind])
        # 指定维度的seq长度符合要求时，将seq加入对应长度范围的ind_data中
        for ind, s in enumerate(splits[:-1]):
            if l >= s and l <= splits[ind + 1]:
                if ind not in ind_data:
                    ind_data[ind] = []
                ind_data[ind].append(x)
                break

    # length分界值
    inds = sorted(list(ind_data.keys()))
    # 每个bucket中的问答对比例
    ind_p = [len(ind_data[x]) / len(all_data) for x in inds]

    if debug:
        print("ind_p: ", np.sum(ind_p), ind_p)

    if isinstance(ws, (list, tuple)):
        assert len(ws) == len(data), "len(ws) 必须等于len(data)，ws是list或者是tuple"

    if isinstance(add_end, bool):
        add_end = [add_end] * len(data)
    else:
        assert(isinstance(add_end, (list, tuple))), "add_end 不是 boolean，就应该是一个list(tuple) of boolean"
        assert len(add_end) == len(data), "如果add_end 是list(tuple)，那么add_end的长度应该和输入数据长度是一致的"

    mul = 2
    if raw:
        mul = 3

    # 生成batches
    while True:
        choice_ind = np.random.choice(inds, p=ind_p)

        if debug:
            print('choice_ind', choice_ind)

        data_batch = random.sample(ind_data[choice_ind], batch_size)
        batches = [[] for i in range(len(data) * mul)]
        max_lens = []
        for j in range(len(data)):
            max_len = max([
                len(x[j]) if hasattr(x[j], '__len__') else 0
                for x in data_batch
            ]) + (1 if add_end[j] else 0)
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws, (list, tuple)):
                    w = ws[j]
                else:
                    w = ws

                # 添加结尾
                line = d[j]
                if add_end[j] and isinstance(line, (tuple, list)):
                    line = list(line) + [WordSequence.END_TAG]

                if w is not None:
                    x, xl = transform_sentence(line, w, max_lens[j], add_end[j])
                    batches[j * mul].append(x)
                    batches[j * mul + 1].append(xl)
                else:
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)
                if raw:
                    batches[j * mul + 2].append(line)

        batches = [np.asarray(x) for x in batches]

        yield batches


def test_batch_flow():
    from fake_data import generate
    x_data, y_data, ws_input, ws_target = generate(size=10000)
    flow = batch_flow([x_data, y_data], [ws_input, ws_target], 4)
    x, xl, y, yl = next(flow)
    print(x.shape, y.shape, xl.shape, yl.shape)


def test_batch_flow_bucket():
    from fake_data import generate
    x_data, y_data, ws_input, ws_target = generate(size=10000)
    flow = batch_flow_bucket([x_data, y_data], [ws_input, ws_target], 4, debug=True)
    for _ in range(10):
        x, xl, y, yl = next(flow)
        print(x.shape, y.shape, xl.shape, yl.shape)


if __name__ == '__main__':
    # print(_get_available_gpus())
    # test_batch_flow()
    test_batch_flow_bucket()