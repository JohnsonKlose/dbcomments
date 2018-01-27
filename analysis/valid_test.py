# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from random import randint

from gensim.models import Word2Vec

from feature_word2vec import getvalue_valid

# 批处理大小
batchSize = 48
# LSTM的单元个数
lstmUnits = 128
# 分类类别
numClasses = 5
# 迭代次数
iterations = 100
# 最大句子长度
maxSeqLength = 60
# 词向量维度
numDimensions = 300

ids_valid = np.load('idsMatrix_valid.npy')
X_valid = getvalue_valid()

mod = Word2Vec.load("Word60/Word60.model")

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(mod.syn0, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0])-1)
predication = tf.matmul(last, weight) + bias

correctPred = tf.equal(tf.argmax(predication, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

