# -*- coding: UTF-8 -*-

import jieba.analyse
import sys

sys.path.append('./')


def cutwords(text):
    data = jieba.cut(text)
    data = [word.encode('utf-8') for word in list(data)]
    stoplist = {}.fromkeys([line.strip() for line in open("/Users/yifengjiao/PycharmProjects/dbcomments/stopwords.txt")])

    segs = [word for word in list(data) if word not in stoplist]
    return segs

if __name__ == "__main__":
    print cutwords('江苏省江阴市黄山路大桥绿洲')