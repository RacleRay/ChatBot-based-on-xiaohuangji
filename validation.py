#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Raclerrr
# datetime:2019/2/3 11:45
# software:PyCharm
# modulename:test

import sys
import pickle

import numpy as np
import tensorflow as tf
from flask import Flask,request


def test(params):

    from seq_2_seq import SequenceToSequence
    from data_utils import batch_flow

    x_data, _ = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))

    # 查看前5条x
    # for x in x_data[:5]:
    #     print(' '.join(x))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0}, # 选择设备的显示格式
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './model/s2ss_chatbot.ckpt'

    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=200,
        **params
    )
    init = tf.global_variables_initializer()


    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            # 本地：
            user_text = input('请输入您的句子:')
            if user_text in ('exit', 'quit'):
                exit(0)
            x_test = [list(user_text.lower())]
            # 接收infos：
            # x_test = [list(infos.lower())]
            bar = batch_flow([x_test], ws, 1)
            x, xl = next(bar)
            x = np.flip(x, axis=1)

            print('输入句子、长度: ', x, xl)

            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print('预测结果ids: ', pred)

            # 转换为words
            print(ws.inverse_transform(x[0]))
            # for p in pred:
            #     ans = ws.inverse_transform(p)
            #     print('预测结果words: ', ans)
            print('预测结果words: ', ws.inverse_transform(pred[0]))
                # return ans


app = Flask(__name__)

@app.route('/api/chatbot', methods=['get'])
def chatbot():
    """flask api"""
    infos = request.args['infos']

    import json
    text = test(json.load(open('params.json')), infos)
    # return text
    # 需要返回字符串，传递给前端
    return ''.join(text)


def main():
    """本地测试"""
    import json
    test(json.load(open('params.json')))


if __name__ == '__main__':
    main()
    # app.debug=True
    # app.run(host='0.0.0.0', port=8000)
    # 预测服务较慢，简化参数或者使用tensorflow提供的方法，可以提高预测速度