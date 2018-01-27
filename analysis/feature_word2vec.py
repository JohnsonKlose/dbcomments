# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('../')

from gensim.models import Word2Vec
import cutwords
import csv
import numpy as np

mod = Word2Vec.load("Word60/Word60.model")
	

def getvalue(text):
    if "很差" == text:
        return 1
    elif "较差" == text:
        return 2
    elif "还行" == text:
        return 3
    elif "推荐" == text:
        return 4
    elif "力荐" == text:
        return 5


def make_sequence_matrix(number, comment_list, filename):
    maxSeqLength = 60
    matrix = np.zeros((number, maxSeqLength), dtype='int32')
    commentCounter = 0
    for comment in comment_list:
        indexCounter = 0
        print str(commentCounter) + ":"
        for word in comment:
            try:
                matrix[commentCounter][indexCounter] = mod.index2word.index(word)
                print matrix[commentCounter][indexCounter]
            except ValueError:
                matrix[commentCounter][indexCounter] = len(mod.index2word)-1
                print str(len(mod.index2word)-1)
            indexCounter += 1
            if indexCounter >= maxSeqLength:
                break
        commentCounter += 1
    print matrix
    np.save(filename, matrix)

csvfile_dunkrik = '../comments/dbcomments_dunkrik.csv'
csvread_dunkrik = csv.reader(open(csvfile_dunkrik))
corpus = []
value = []
for row in csvread_dunkrik:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus.append(row[-1])
    value.append(getvalue(row[2]))

csvfile_fanghua = '../comments/dbcomments_fanghua.csv'
csvread_fanghua = csv.reader(open(csvfile_fanghua))
for row in csvread_fanghua:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus.append(row[-1])
    value.append(getvalue(row[2]))

csvfile_fengshenchuanqi = '../comments/dbcomments_fengshenchuanqi.csv'
csvread_fengshenchuanqi = csv.reader(open(csvfile_fengshenchuanqi))
for row in csvread_fengshenchuanqi:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus.append(row[-1])
    value.append(getvalue(row[2]))

csvfile_fuchunshanjutu = '../comments/dbcomments_fengshenchuanqi.csv'
csvread_fuchunshanjutu = csv.reader(open(csvfile_fuchunshanjutu))
for row in csvread_fuchunshanjutu:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus.append(row[-1])
    value.append(getvalue(row[2]))

csvfile_jianianhua = '../comments/dbcomments_jianianhua.csv'
csvread_jianianhua = csv.reader(open(csvfile_jianianhua))
for row in csvread_fuchunshanjutu:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus.append(row[-1])
    value.append(getvalue(row[2]))

csvfile_qianren3 = '../comments/dbcomments_qianren3.csv'
csvread_qianren3 = csv.reader(open(csvfile_qianren3))
for row in csvread_qianren3:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus.append(row[-1])
    value.append(getvalue(row[2]))

csvfile_qinghenangao = '../comments/dbcomments_qinghenangao.csv'
csvread_qinghenangao = csv.reader(open(csvfile_qinghenangao))
for row in csvread_qinghenangao:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus.append(row[-1])
    value.append(getvalue(row[2]))

csvfile_sanshengsanshi = '../comments/dbcomments_sanshengsanshi.csv'
csvread_sanshengsanshi = csv.reader(open(csvfile_sanshengsanshi))
for row in csvread_sanshengsanshi:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus.append(row[-1])
    value.append(getvalue(row[2]))

csvfile_zhianshike = '../comments/dbcomments_zhianshike.csv'
csvread_zhianshike = csv.reader(open(csvfile_zhianshike))
for row in csvread_zhianshike:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus.append(row[-1])
    value.append(getvalue(row[2]))

numWords = []
numCount = []
for index, val in enumerate(corpus):
    numWords.append(cutwords.cutwords(corpus[index]))
    numCount.append(len(numWords[index]))
print numWords[0], len(numCount)
print ('The total number of comments is', len(numCount))
print ('The total number of words in the files is', sum(numCount))
print ('The average number of words in the files is', sum(numCount)/len(numCount))

# make_sequence_matrix(len(numCount), numWords, 'idsMatrix')


def getvalue_train():
    return value

corpus_valid = []
value_valid = []
csvfile_wukongzhuan = '../comments/dbcomments_wukongzhuan.csv'
csvread_wukongzhuan = csv.reader(open(csvfile_wukongzhuan))
for row in csvread_wukongzhuan:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus_valid.append(row[-1])
    value_valid.append(getvalue(row[2]))

csvfile_yinianwuming = '../comments/dbcomments_yinianwuming.csv'
csvread_yinianwuming = csv.reader(open(csvfile_yinianwuming))
for row in csvread_yinianwuming:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus_valid.append(row[-1])
    value_valid.append(getvalue(row[2]))

csvfile_tiancaiqiangshou = '../comments/dbcomments_tiancaiqiangshou.csv'
csvread_tiancaiqiangshou = csv.reader(open(csvfile_tiancaiqiangshou))
for row in csvread_yinianwuming:
    if row[2] == "None":
        continue
    if row[2] == "评星":
        continue
    corpus_valid.append(row[-1])
    value_valid.append(getvalue(row[2]))

numWords_valid = []
numCount_valid = []
for index, val in enumerate(corpus_valid):
    numWords_valid.append(cutwords.cutwords(corpus_valid[index]))
    numCount_valid.append(len(numWords_valid[index]))
print numWords_valid[0], len(numCount_valid)

# make_sequence_matrix(len(numCount_valid), numWords_valid, 'idsMatrix_valid')


def getvalue_valid():
    return value_valid


