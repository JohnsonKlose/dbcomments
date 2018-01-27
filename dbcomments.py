# coding=utf-8
import requests
from bs4 import BeautifulSoup
import csv
import sys
import codecs
import time

reload(sys)
sys.setdefaultencoding('utf-8')

# 《敦刻尔克》豆瓣评论地址
url = 'https://movie.douban.com/subject/26607693/comments?start=0'
head = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'}
cookies = {'cookie': 'll="118159"; bid=_vKZDI27ugI; ps=y; ue="johnsonzn@163.com"; dbcl2="63947146:jdOBTMY5NsQ"; _ga=GA1.2.790170630.1503927208; _gid=GA1.2.78626124.1504017107; ck=tuXF; _vwo_uuid_v2=77BAF1538510F4AEC8AA6A43453D0C2F|1a286c3eb772417b868dd7c3f89923be; ap=1; push_noty_num=0; push_doumail_num=0; __utmt=1; __utma=30149280.790170630.1503927208.1504099070.1504101681.10; __utmb=30149280.2.10.1504101681; __utmc=30149280; __utmz=30149280.1504099070.9.6.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utmv=30149280.6394'}

filename = '/Users/yifengjiao/PycharmProjects/dbcomments/dbcomments.csv'
f = open(filename, 'w')
f.write(codecs.BOM_UTF8)
writer = csv.writer(f)
writer.writerow(['是否看过', '日期', '评星', '赞成数', '评论内容'])

# 请求网站
html = requests.get(url, headers=head, cookies=cookies)

while html.status_code == 200:

    # 生成BeautifulSoup对象
    soup = BeautifulSoup(html.text, 'html.parser')
    comment = soup.find_all(class_='comment')

    # 解析每一个class为comment的div中的内容
    for com in comment:

        # 评论内容
        commentstring = com.p.string
        print commentstring

        # 评星(会出现没有评星的情况,没有评星设置为None)
        rating = com.find(class_='rating')
        if rating != None:
            ratingstring = rating.get('title')
        else:
            ratingstring = 'None'
        print ratingstring

        # 评论有用的数量
        votes = com.find(class_='votes')
        votesstring = votes.string
        print votesstring

        # 是否看过
        commentinfo = com.find(class_='comment-info')
        lookstring = commentinfo.span.text
        print lookstring

        # 日期
        commenttime = com.find(class_='comment-time')
        timestring = commenttime.get('title')
        print timestring

        # 写入csv文件一行数据
        try:
            writer.writerow([lookstring, timestring, ratingstring, votesstring, commentstring])
        except Exception as err:
            print err

    time.sleep(2)
    # 下一页
    nextstring = soup.find(class_='next').get('href')
    nexturl = 'https://movie.douban.com/subject/26607693/comments' + nextstring
    html = requests.get(nexturl, headers=head, cookies=cookies)

f.close()