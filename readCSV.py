# coding=utf-8
import csv

def readCSVByRow(filename, rowname):
    with open(filename) as csvfile:
        readerRow = csv.DictReader(csvfile)
        list = []
        for row in readerRow:
            list.append(row[rowname])
        return list

'''输入csv某一列的某值,返回等于该值的数量'''
def returnValueLengthByRowValue(filename, rowname, value):
    with open(filename) as csvfile:
        readerRow = csv.DictReader(csvfile)
        count = 0
        for row in readerRow:
            if row[rowname] == value:
                count = count + 1
        return count

'''得到csv所有数据数量'''
def getdatacount(filename):
    with open(filename) as csvfile:
        readerRow = csv.DictReader(csvfile)
        count = 0
        for row in readerRow:
            count = count + 1
        return count

'''用list格式返回所有数据'''
def getdatarow(filename):
    reader = csv.reader(open(filename))
    list = []
    for row in reader:
        list.append(row)
    return list



# readlist = readCSVByRow("/Users/yifengjiao/Documents/dataFile/ldcompletetest.csv", "GANHAO")
# print readlist

# rowValueLength = returnValueLengthByRowValue("/Users/yifengjiao/Documents/dataFile/ldcomplete_UTF-8.csv", "TYPE", "功能灯")
# print rowValueLength


# with open("/Users/yifengjiao/Documents/dataFile/ldcompletetest.csv") as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         print (row["GANHAO"])
