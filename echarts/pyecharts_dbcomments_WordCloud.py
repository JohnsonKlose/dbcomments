import cutwords
from collections import Counter
import csv
import pandas as pd
from pyecharts import WordCloud

csvfile = '../dbcomments.csv'
csv_read = csv.reader(open(csvfile))
text = ""
for row in csv_read:
    text += row[-1]
cutdata = cutwords.cutwords(text)
worddata = dict(Counter(cutdata))

# csvdata = pd.read_csv(csvfile)
# content_col = csvdata.iloc[:, 4]
# content = content_col.values
# content_text = ""
# for k in range(0, len(content)):
#     content_text += str(content[k]).strip()
# cutdata = cutwords.cutwords(content_text)
# worddata = dict(Counter(cutdata))

word = []
value = []
print len(worddata)
for (k, v) in worddata.items():
    if v < 50:
        del worddata[k]
    else:
        word.append(k)
        value.append(v)
print len(worddata)
wordcloud = WordCloud(width=1300, height=620)
wordcloud.add("", word, value, word_size_range=[20, 1000], shape='diamond')
wordcloud.render("../visualization/render_dbcomments_wordcloud.html")
