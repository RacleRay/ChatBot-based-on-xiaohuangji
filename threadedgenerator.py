#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Raclerrr
# datetime:2019/1/24 22:52
# software:PyCharm
# modulename:threadedgenerator
from threading import Thread
from queue import Queue

class ThreadedGenerator(object):
    """线程生成类，可实现多线程io，同时不在主线程内进行，防止主线程卡顿"""
    def __init__(self, iterator,
                 sentinel=object(),
                 queue_maxsize=0,
                 daemon=False):
        self._iterator = iterator  # 传入可迭代对象
        self._sentinel = sentinel  # 指示对象
        self._queue = Queue(maxsize=queue_maxsize)  # 新建队列
        self._thread = Thread(     # 新建线程
            name=repr(iterator),
            target=self._run       # 执行对象
        )
        self._thread.daemon = daemon  # 是否为守护线程
        self._started = False      # 是否开始迭代生成

    def __repr__(self):
        # In str.format, !s chooses to use str to format the object whereas
        # !r chooses repr to format the value.
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        """开启线程"""
        try:
            for value in self._iterator:
                if not self._started:
                    return
                self._queue.put(value)
        finally:
            self._queue.put(self._sentinel) # 指示对象

    def close(self):
        """关闭线程"""
        self._started = False
        try:
            while True:
                self._queue.get(timeout=30)
        except KeyboardInterrupt as e:
            raise e
        except:
            pass

    def __iter__(self):
        """重写迭代函数"""
        self._started = True
        self._thread.start()
        # the callable is called until it returns the sentinel.
        for value in iter(self._queue.get, self._sentinel):
            yield value
        self._thread.join()
        self._started = False

    def __next__(self):
        if not self._started:
            self._started = True
            self._thread.start()
        value = self._queue.get(timeout=30)
        if value == self._sentinel:
            raise StopIteration()
        return value


def test():

    def gene():
        i = 0
        while True:
            yield i
            i += 1

    t = gene()
    test = ThreadedGenerator(t)

    for _ in range(10):
        print(next(test))

    test.close()


if __name__ == '__main__':
    test()