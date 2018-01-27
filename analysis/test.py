# -*- coding: UTF-8 -*-
import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

maxSeqLength = 10
numDimensions = 300
sentence = np.zeros(maxSeqLength, dtype='int32')
mod = Word2Vec.load("Word60.model")
sentence[0] = mod.index2word.index("我")
sentence[1] = mod.index2word.index("很")
sentence[2] = mod.index2word.index("喜欢")
sentence[3] = mod.index2word.index("这部")
sentence[4] = mod.index2word.index("电影")
sentence[5] = mod.index2word.index("非常")
sentence[6] = mod.index2word.index("好看")
print(sentence.shape)
print(sentence)

with tf.Session() as sess:
    print (tf.nn.embedding_lookup(mod.syn0, sentence).eval().shape)
