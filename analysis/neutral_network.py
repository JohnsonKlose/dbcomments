# -*- coding: UTF-8 -*-
import sys
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
import datetime
from random import randint
from feature_word2vec import getvalue_train
from feature_word2vec import getvalue_valid

sys.path.append('./')

# 记录运行开始时间
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 加载训练数据
ids = np.load('idsMatrix.npy')
# 加载训练标记
X = getvalue_train()
# 加载验证数据
ids_valid = np.load('idsMatrix_valid.npy')
# 加载验证标记
X_valid = getvalue_valid()


# 生成训练集Batch
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for inx in range(batchSize):
        num = randint(1, len(ids)-1)
        print num
        if X[num] == 1:
            labels.append([1, 0, 0, 0, 0])
        elif X[num] == 2:
            labels.append([0, 1, 0, 0, 0])
        elif X[num] == 3:
            labels.append([0, 0, 1, 0, 0])
        elif X[num] == 4:
            labels.append([0, 0, 0, 1, 0])
        elif X[num] == 5:
            labels.append([0, 0, 0, 0, 1])
        arr[inx] = ids[num-1:num]
        print arr[inx], labels[inx]
    return arr, labels


# 生成验证集Batch
def getValidBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for inx in range(batchSize):
        num = randint(1, len(ids_valid)-1)
        # print num
        if X_valid[num] == 1:
            labels.append([1, 0, 0, 0, 0])
        elif X_valid[num] == 2:
            labels.append([0, 1, 0, 0, 0])
        elif X_valid[num] == 3:
            labels.append([0, 0, 1, 0, 0])
        elif X_valid[num] == 4:
            labels.append([0, 0, 0, 1, 0])
        elif X_valid[num] == 5:
            labels.append([0, 0, 0, 0, 1])
        arr[inx] = ids_valid[num-1:num]
        # print arr[inx], labels[inx]
    return arr, labels

# 批处理大小
batchSize = 48
# LSTM的单元个数
lstmUnits = 128
# 分类类别
numClasses = 5
# 迭代次数
iterations = 100000
# 最大句子长度
maxSeqLength = 60
# 词向量维度
numDimensions = 300

# 加载词向量模型
mod = Word2Vec.load("Word60/Word60.model")

tf.reset_default_graph()

# 生成标签占位符
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
# 生成输入数据占位符
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

# 调用tf.nn.embedding_lookup()函数返回词向量
# 第一个维度是批处理大小, 第二个维度是句子长度, 第三个维度是词向量长度
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(mod.syn0, input_data)

# 构建LSTM cell
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
# dropout参数, 避免过拟合
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
# 将LSTM cell和三维数据输入到tf.nn.dynamic_rn, 构建整个网络
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

# 权重矩阵
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
# 偏置项
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
# 处理value值
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0])-1)
# 获得最后的输出值
predication = tf.matmul(last, weight) + bias

# 定义预测函数
correctPred = tf.equal(tf.argmax(predication, 1), tf.argmax(labels, 1))
# 定义准确率
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
# 定义标准的交叉熵损失函数来作为损失值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predication, labels=labels))
# 定义Adam优化器
optimizer = tf.train.AdamOptimizer().minimize(loss)

# 将损失值和准确率加入到Tensorboard中
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)

# 定义一个TensorFlow会话, 并开始训练
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# 开始迭代的过程
for i in range(iterations):
    # 生成下一个Batch
    nextBatch, nextBatchLabels = getTrainBatch()
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    # 将summary写入Tensorboard
    if i % 50 == 0:
        nextBatch_valid, nextBatchLabels_valid = getValidBatch()
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    # 每10000次保存
    if i % 10000 == 0 and i != 0:
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print "saved to %s" % save_path
writer.close()

end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

print "start time :" + str(start_time)
print "end time :" + str(end_time)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

iteration = 10
for i in range(iteration):
    next_batch_valid, next_batch_labels_valid = getValidBatch()
    print ("Accuracy for this batch:", (sess.run(accuracy, {input_data: next_batch_valid,
                                                            labels: next_batch_labels_valid})))