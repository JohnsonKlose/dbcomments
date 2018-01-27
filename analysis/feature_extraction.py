# -*- coding: UTF-8 -*-
import csv
import cutwords
from sklearn.feature_extraction.text import TfidfVectorizer


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

csvfile = '../dbcomments.csv'
csv_read = csv.reader(open(csvfile))
corpus = ""
value = []
for row in csv_read:
    if row[2] == "None":
        continue
    corpus += (row[-1])
    value.append(getvalue(row[2]))
cutdata = cutwords.cutwords(corpus)
tfidfvectorizer = TfidfVectorizer()
X = tfidfvectorizer.fit_transform(cutdata)


def getfeature():
    return X


def getvalue():
    return value