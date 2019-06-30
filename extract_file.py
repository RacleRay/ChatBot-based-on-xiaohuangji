#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:Raclerrr
# datetime:2019/1/21 17:48
# software:PyCharm
# modulename:extract_file
import re
import sys
import pickle
from tqdm import tqdm


def make_split(line):
    """合并句子时使用的方法：匹配常用的符号，匹配成功返回空，不成功使用', '做切分"""
    if re.match(r'.*([，…?!\.,!？])$', ''.join(line)):
        return []
    return [', ']


def good_line(line):
    """判断是否是有效的句子，排除无用的字母和数字，比如：url"""
    if len(line) == 0:
        return False
    ch_count = 0
    for c in line:
        # 中文字符范围
        if '\u4e00' <= c <= '\u9fff':
            ch_count += 1
    if ch_count / float(len(line)) >= 0.8 and len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) < 3 \
            and len(re.findall(r'[ˇˊˋˍεπのゞェーω]', ''.join(line))) < 3:
        return True
    return False


def regular(sen):
    """替换连续出现的符号，英文符号替换为中文符号"""
    sen = re.sub(r'\.{3, 100}', '…', sen)
    sen = re.sub(r'…{2,100}', '…', sen)
    sen = re.sub(r'[,]{1,100}', '，', sen)
    sen = re.sub(r'[\.]{1,100}', '。', sen)
    sen = re.sub(r'[\?]{1,100}', '？', sen)
    sen = re.sub(r'[!]{1,100}', '！', sen)
    sen = re.sub(r'…{1,100}', '…', sen)
    sen = re.sub(r'\.{3,100}', '…', sen)
    sen = re.sub(r'···{2,100}', '…', sen)
    sen = re.sub(r',{1,100}', '，', sen)
    sen = re.sub(r'\.{1,100}', '。', sen)
    sen = re.sub(r'。{1,100}', '。', sen)
    sen = re.sub(r'\?{1,100}', '？', sen)
    sen = re.sub(r'？{1,100}', '？', sen)
    sen = re.sub(r'!{1,100}', '！', sen)
    sen = re.sub(r'！{1,100}', '！', sen)
    sen = re.sub(r'~{1,100}', '～', sen)
    sen = re.sub(r'～{1,100}', '～', sen)
    sen = re.sub(r'[“”]{1,100}', '"', sen)
    sen = re.sub('[^\w\u4e00-\u9fff"。，？！～·]+', '', sen)
    sen = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', sen)
    return sen


def main(limit=20, x_limit=1, y_limit=2):
    """文本数据处理main函数
    args: limit——问句和答句的最大长度；
          x_limit——问句的最小长度；
          y_limit——答句的最小长度
    """
    from word_sequence import WordSequence

    print('extracting lines')
    fp = open("xiaohuangji50w_fenciA.conv",
              'r', errors='ignore', encoding='utf-8')
    groups = []
    group = []

    # 文本规范化
    for line in tqdm(fp):
        if line.startswith('M '):
            line = line.replace('\n', '') # 去除回车符
            # 查看文本文件，具体情况具体设计
            if '/' in line:
                line = line[2:].split('/')
            else:
                line = list(line[2:])
            line = line[:-1]
            group.append(list(regular(''.join(line))))
        else:
            if group:
                groups.append(group)
                group = []
    # 最后一次加入的group，在M 开头行后没有E 开头的下一行，退出循环，group有值
    if group:
        groups.append(group)
        del group

    # 问答对处理（文件为电影台词）
    x_data = [] # 问
    y_data = [] # 答
    for group in tqdm(groups):
        for i, line in enumerate(group):
            # pre_line：group多于两行时，i行的前一行
            pre_line = None
            if i > 0:
                pre_line = group[i - 1]
                if not good_line(pre_line):
                    last_line = None
            # next_line：第i行在倒数第一行之前时，下一行
            next_line = None
            if i < len(group) - 1:
                next_line = group[i + 1]
                if not good_line(next_line):
                    next_line = None
            # next_next_line ：第i行在倒数第二行之前时，下一行的下一行
            next_next_line = None
            if i < len(group) - 2:
                next_next_line = group[i + 2]
                if not good_line(next_next_line):
                    next_next_line = None

            # 如果当前行的下一行存在，i行加入x_data问句，i+1行加入y_data答句
            if next_line:
                x_data.append(line)
                y_data.append(next_line)
            # 如果当前行的上一行和下一行都存在，i-1和i行加入x_data问句，i+1行加入y_data答句
            # make_split：合并两句为一个序列所采用的分隔符
            if pre_line and next_line:
                x_data.append(pre_line + make_split(pre_line) + line)
                y_data.append(next_line)
            # 如果当前行的下一行和下下一行都存在，i行加入x_data问句，i+1和i+2行加入y_data答句
            if next_line and next_next_line:
                x_data.append(line)
                y_data.append(next_line + make_split(next_line) + next_next_line)

    print(len(x_data), len(y_data))

    # 打印前20组问答对结果
    for ask, answer in zip(x_data[:20], y_data[:20]):
        print(''.join(ask))
        print(''.join(answer))
        print('-' * 20)

    data = list(zip(x_data, y_data))
    data = [(x, y)
            for x, y in data
            if len(x) < limit
            and len(y) < limit
            and len(y) >= y_limit
            and len(x) >= x_limit]

    # 长度筛选之后的问答句子
    x_data, y_data = zip(*data)

    # 对输入的语料进行词统计，生成词表
    print('fit word_sequence')
    ws_input = WordSequence()
    ws_input.fit(x_data + y_data)

    # 保存问答句子和词表
    print('dump')
    # 打包保存问答句子
    pickle.dump(
        (x_data, y_data),
        open('chatbot.pkl', 'wb')
    )
    # 打包保存词表
    pickle.dump(ws_input, open('ws.pkl', 'wb'))

    print('done')


if __name__ == '__main__':
    main()