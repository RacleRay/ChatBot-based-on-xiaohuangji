#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Raclerrr
# datetime:2019/2/2 23:03
# software:PyCharm
# modulename:train
import sys
import random
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def train(params):
    from seq_2_seq import SequenceToSequence
    from data_utils import batch_flow_bucket as batch_flow
    from threadedgenerator import ThreadedGenerator

    x_data, y_data = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))


    # 训练模式
    n_epoch = 200
    batch_size = 256

    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        allow_soft_placement=True, # 系统自动选择运行cpu或者gpu
        log_device_placement=False # 是否需要打印设备日志
    )

    save_path = './model/s2ss_chatbot.ckpt'

    # 重置默认的图
    tf.reset_default_graph()
    # 定义图的基本信息
    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.Session(config=config) as sess:
            # 定义模型
            model = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=batch_size,
                **params
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            if tf.train.checkpoint_exists('./model/s2ss_chatbot.ckpt'):
                model.load(sess, save_path)
                print('>>>=Having restored model')

            flow = ThreadedGenerator(
                batch_flow([x_data, y_data],
                           ws,
                           batch_size,
                           add_end=[False, True]),
                queue_maxsize=30
            )

            for epoch in range(1, n_epoch +1):
                costs = []
                bar = tqdm(range(steps),
                           total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = next(flow)
                    # 此处效果为每个seq倒序
                    x = np.flip(x, 1)
                    cost, lr = model.train(sess, x, xl, y, yl, return_lr=True)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(
                        epoch,
                        np.mean(costs),
                        lr
                    ))

                model.save(sess, save_path)
            flow.close()


    # 测试模式
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=200,
        parallel_iterations=1,
        **params
    )

    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        bar = batch_flow([x_data, y_data], ws, 1, add_end=False)
        t = 0
        for x, xl, y, yl in bar:
            x = np.flip(x, axis=1)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(ws.inverse_transform(x[0]))
            print(ws.inverse_transform(y[0]))
            print(ws.inverse_transform(pred[0]))
            t += 1
            # 此处只测试了3次
            if t >= 3:
                break


def main():
    import json
    train(json.load(open('params.json')))


if __name__ == '__main__':
    main()


