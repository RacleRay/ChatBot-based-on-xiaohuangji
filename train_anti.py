#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Raclerrr
# datetime:2019/2/3 11:12
# software:PyCharm
# modulename:trian_anti
import random
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def train(params):

    from seq_2_seq import SequenceToSequence
    from data_utils import batch_flow_bucket as batch_flow
    from word_sequence import WordSequence
    from threadedgenerator import ThreadedGenerator

    x_data, y_data = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))

    n_epoch = 40
    batch_size = 128
    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = 'model_anti/s2ss_chatbot_anti.ckpt'
    best_save_path = 'model_anti_best/best_cost.ckpt'

    # 训练模式
    # loss下降较慢，不至于出现严重的梯度消散
    tf.reset_default_graph()
    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.Session(config=config) as sess:
            model = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=batch_size,
                **params
            )
            init = tf.global_variables_initializer()
            sess.run(init)
            
            # 是否继续训练
            if tf.train.checkpoint_exists('./model_anti/s2ss_chatbot_anti.ckpt'):
                model.load(sess, save_path)
                print('>>>=Having restored model')

            flow = ThreadedGenerator(
                batch_flow([x_data, y_data], ws, batch_size, add_end=[False, True]),
                queue_maxsize=30
            )
            
            dummy_encoder_inputs = np.array([
                np.array([WordSequence.PAD]) for _ in range(batch_size)])
            dummy_encoder_inputs_length = np.array([1] * batch_size)
            
            temp_loss = 30
            for epoch in range(1, n_epoch + 1):
                costs = []
                bar = tqdm(range(steps), total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = next(flow)
                    # permutation = np.random.permutation(batch_size)
                    # dummy_encoder_inputs = x[permutation, :]
                    # dummy_encoder_inputs_length = xl[permutation]
                    x = np.flip(x, axis=1)
                    dummy_encoder_inputs = np.flip(dummy_encoder_inputs, axis=1)

                    add_loss = model.train(sess,
                                           dummy_encoder_inputs,
                                           dummy_encoder_inputs_length,
                                           y, yl, loss_only=True)
                    add_loss *= -0.5  # 此处相当于减去加入负样本所带来的损失

                    cost, lr = model.train(sess, x, xl, y, yl,
                                           return_lr=True, add_loss=add_loss)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(
                        epoch,
                        np.mean(costs),
                        lr
                    ))
                model.save(sess, save_path)
                
                mean_loss = np.mean(costs)
                if mean_loss <= temp_loss:
                    model.save(sess, best_save_path)
                    temp_loss = mean_loss
                
                with open('./model_anti/globalstep.txt', 'a+') as f:
                    f.write('global step is:{}\n'.format(epoch))
                    
            flow.close()

    # 预测模式（beam_width=200）
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

        bar = batch_flow([x_data, y_data], ws, 1, add_end=False)
        t = 0
        for x, xl, y, yl in bar:
            x = np.flip(x, axis=1)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print('输入问句（倒序）：', ws.inverse_transform(x[0]))
            print('输入答句：', ws.inverse_transform(y[0]))
            print('预测答句：', ws.inverse_transform(pred[0][0]))
            t += 1
            if t >= 3:
                break

    # 预测模式（beam_width=1）
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=1,
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
            print('输入问句（倒序）：', ws.inverse_transform(x[0]))
            print('输入答句：', ws.inverse_transform(y[0]))
            print('预测答句：', ws.inverse_transform(pred[0][0]))
            t += 1
            if t >= 3:
                break


def main():
    import json
    train(json.load(open('params.json')))


if __name__ == '__main__':
    main()