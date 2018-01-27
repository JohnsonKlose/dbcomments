# coding=utf-8
from pyecharts import Line
import readCSV

csvFilename = "../dbcomments.csv"
csvRowName = '评星'

ratingtype = ['None', '很差', '较差', '还行', '推荐', '力荐']
ratingtypevalue = []
for type in ratingtype:
    value = readCSV.returnValueLengthByRowValue(csvFilename, csvRowName, type)
    ratingtypevalue.append(value)

line = Line('评分折线图')
line.add('《敦刻尔克》', ratingtype, ratingtypevalue, is_smooth=True, mark_line=["max"])
line.render('../visualization/render_dbcomments_line.html')
