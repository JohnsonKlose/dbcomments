# -*- coding: UTF-8 -*-
from pyecharts import Bar
import readCSV

csvFilename = "../dbcomments.csv"
csvRowName = '评星'

ratingtype = ['None', '很差', '较差', '还行', '推荐', '力荐']
ratingtypevalue = []
for type in ratingtype:
    value = readCSV.returnValueLengthByRowValue(csvFilename, csvRowName, type)
    ratingtypevalue.append(value)

bar = Bar("柱状图", "不同评分类型数量")
bar.add("评分类型", ratingtype, ratingtypevalue)
bar.render("../visualization/render_dbcomments_bar.html")
