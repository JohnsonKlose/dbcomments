# coding=utf-8
from pyecharts import Pie
import readCSV

csvFilename = "../dbcomments.csv"
csvRowName = '评星'

ratingtype = ['None', '很差', '较差', '还行', '推荐', '力荐']
ratingtypevalue = []
for type in ratingtype:
    value = readCSV.returnValueLengthByRowValue(csvFilename, csvRowName, type)
    ratingtypevalue.append(value)

pie = Pie("评分饼状图")
pie.add("玫瑰饼状图", ratingtype, ratingtypevalue, is_random=True, radius=[30, 75],
        is_legend_show=True, is_label_show=True)
pie.show_config()
pie.render("../visualization/render_dbcomments_pie.html")