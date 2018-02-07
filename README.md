# dbcomments
此项目爬取豆瓣电影的评论信息，包括评论内容、评星、评论有用的数量、是否看过和日期等信息，对爬取的评论信息进行可视化处理，同时运用循环神经网络和随机森林两种机器学习算法，构建基于评论内容预测评星等级的模型。  

## 爬虫思路
- 分析豆瓣评论地址url：  
查看到豆瓣电影评论的访问地址为  
[https://movie.douban.com/subject/26628329/comments?start=0&limit=20&sort=new_score&status=P&percent_type=](https://movie.douban.com/subject/26628329/comments?start=0&limit=20&sort=new_score&status=P&percent_type=)  
分析url我们可以看到，subject/后面的一串数字代表着这个电影的编号，start参数表示开始评论数，0就代表这是第一页，评论从第1条开始，limit代表每页显示的评论数
- 分析页面DOM  
在Chrome中查看页面的DOM结构，分析评论内容、评星、评论有用数量、是否看过和日期信息可以通过什么方式爬取，本项目采用BeautifulSoap框架对DOM进行解析，下面将对这些数据的爬取方式一一讲解。
	- 所有的评论详细内容，包括评星等信息，都在class为comment的标签下，如下图所示，因此第一步就是找到所有class为comment的标签作为主节点：  
	```
	comment = soup.find_all(class_='comment')
	```  
	![comment](http://oswrmk9hd.bkt.clouddn.com/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202018-02-03%20%E4%B8%8A%E5%8D%8811.42.37.png)  
	- 评论内容在主节点的直接孩子节点<p>节点中：  
	```
    commentstring = com.p.string
	```  
	![commentstring](http://oswrmk9hd.bkt.clouddn.com/4577C96F-E181-4157-904E-21B65148BBFD.png)  
	- 评星的内容在class为rating的节点中的title标签下：  
	```
	ratingstring = rating.get('title')
	```  
	![rating](http://oswrmk9hd.bkt.clouddn.com/1448F836-1575-4D1D-9664-DC011ED1164C.png)  
	- 评论有用的数量在class为votes的节点中：  
	```
	votes = com.find(class_='votes')
	votesstring = votes.string
	```  
	![votes](http://oswrmk9hd.bkt.clouddn.com/FA5F0654-AB93-4D8E-8819-502FE71C6155.png)  
	- 是否看过的信息在	class为comment-info的节点下第一个标签为<span>的孩子节点中：    
	```
	commentinfo = com.find(class_='comment-info')
    lookstring = commentinfo.span.text
	```
	![look](http://oswrmk9hd.bkt.clouddn.com/87BFD8DA-877D-4671-B072-DB74BE45D926.png)  
	- 日期信息在class为comment-time节点中的title标签下：  
	```
	commenttime = com.find(class_='comment-time')
	timestring = commenttime.get('title')
	```
	![time](http://oswrmk9hd.bkt.clouddn.com/506E181B-426C-46BF-A4D4-C1D3084E9707.png)
- 跳转下一页  
爬取完一个页面的评论之后，需要翻到下一页继续爬取评论，分析页面我们发现，“后页”按钮节点中href标签存放着下页访问的地址信息，因此要构造下一页的url，并访问下一页继续爬取评论信息：  
```
nextstring = soup.find(class_='next').get('href')
nexturl = 'https://movie.douban.com/subject/26607693/comments' + nextstring
html = requests.get(nexturl, headers=head, cookies=cookies)
```  
![next](http://oswrmk9hd.bkt.clouddn.com/4D5D8701-1B5E-4F22-BCA5-0D4E494FFE3C.png)  
- 存放数据  
本项目爬取的数据均存放在csv文件中，便于读取：
```
writer = csv.writer(f)
writer.writerow(['是否看过', '日期', '评星', '赞成数', '评论内容'])
...
try:
    writer.writerow([lookstring, timestring, ratingstring, votesstring, commentstring])
except Exception as err:
    print err
```  
**Tips:** 每次爬取一个页面时，可以暂停2秒，降低爬取的速度。当前豆瓣电影页面中只能显示前500条评论了，因此每部电影只能爬取500条评论了，如果有更好地办法能爬取所有评论的话，可以留言或邮箱与我沟通联系。  

## 数据可视化
本项目采用的是pyecharts这个包，具体可参考[https://github.com/pyecharts/pyecharts](https://github.com/pyecharts/pyecharts)。下面是针对评分信息绘制的柱状图、折线图和饼状图的效果图：  
![bar](http://oswrmk9hd.bkt.clouddn.com/%E6%9F%B1%E7%8A%B6%E5%9B%BE.png)  
![line](http://oswrmk9hd.bkt.clouddn.com/%E8%AF%84%E5%88%86%E6%8A%98%E7%BA%BF%E5%9B%BE.png)  
![pie](http://oswrmk9hd.bkt.clouddn.com/%E8%AF%84%E5%88%86%E9%A5%BC%E7%8A%B6%E5%9B%BE.png)  
词云图是能制作出基于词频数据美观酷炫的可视化效果，本项目基于pyecharts生成了词云图，具体步骤如下：  
- 将所有评论分词  
```
csvfile = '../dbcomments.csv'
csv_read = csv.reader(open(csvfile))
text = ""
for row in csv_read:
	text += row[-1]
cutdata = cutwords.cutwords(text)
worddata = dict(Counter(cutdata))
```  
- 生成dict格式的词频  
```
worddata = dict(Counter(cutdata))
```  
- 生成词列表和词频列表  
```
word = []
value = []
for (k, v) in worddata.items():
    if v < 50:
        del worddata[k]
    else:
        word.append(k)
        value.append(v)
```
- 生成词云图  
```
wordcloud = WordCloud(width=1300, height=620)
wordcloud.add("", word, value, word_size_range=[20, 1000], shape='diamond')
wordcloud.render()
```  
- 效果图如下：  
![wordcloud](http://oswrmk9hd.bkt.clouddn.com/echarts.png)  

## 神经网络
本项目爬取数据中，有评论内容和评级两组数据，考虑评论内容与评级之间有关联，因此建立预测模型通过评论内容自动预测该评论的评级，首先通过循环神经网络(RNN)加长短期记忆网络(LSTM)建立预测模型，具体步骤如下：
- 建立评论的词向量特征  
词向量的意思是将文字通过特定维度的向量来表示，通过量化的特征值可以作为机器学习或者深度学习模型的输入数据。本项目使用的是“Word2Vec”，简单的说，这个模型根据上下文的语境来推断出每个词的词向量。详细内容可以参考Tensorflow中的资料[https://www.tensorflow.org/tutorials/word2vec](https://www.tensorflow.org/tutorials/word2vec)；  
gensim是一个非常好的自然语言处理库，本项目直接使用通过gensim库训练好的词向量模型，存放在项目Word60文件夹中，但是模型文件太大，GitHub不允许上传，所以需要文件的可以前往[https://pan.baidu.com/s/1nvWye2t](https://pan.baidu.com/s/1nvWye2t)地址下载，提取码是ei7t；  
![word60](http://oswrmk9hd.bkt.clouddn.com/7024FB13-EE46-41B3-AEA1-1AAAC1C2046D.png)  
直接加载词向量模型即可使用，代码如下：  
```
from gensim.models import Word2Vec
mod = Word2Vec.load("Word60/Word60.model")
```  
- 训练数据集合验证数据集
通过我们之前爬取的数据建立训练和验证数据集，需要训练样本、训练标签、验证样本、验证标签四类数据。
	- 加载爬取数据csv文件，value为训练标签：  
	```
	corpus = []
	value = []
	csvfile_zhianshike = '../comments/dbcomments_zhianshike.csv'
	csvread_zhianshike = csv.reader(open(csvfile_zhianshike))
	for row in csvread_zhianshike:
    	if row[2] == "None":
        	continue
    	if row[2] == "评星":
        	continue
    	corpus.append(row[-1])
    	value.append(getvalue(row[2]))
	```  
	**Tips:**需要注意的是，我们爬取数据中评星分为5个等级：力荐、推荐、还行、较差、很差，因此我们的分类标签分为5种，神经网络中我们通常通过向量的方式表示分类标签，因此我们将5个等级对应的标签分别设为[1,0,0,0,0][0,1,0,0,0][0,0,1,0,0][0,0,0,1,0][0,0,0,0,1]  
	- 将每条评论进行分词，numWords存放分词结果，numCount存放分词的长度：  
	```
	numWords = []
	numCount = []
	for index, val in enumerate(corpus):
    	numWords.append(cutwords.cutwords(corpus[index]))
    	numCount.append(len(numWords[index]))
    print ('The total number of comments is', len(numCount))
	print ('The total number of words in the files is', sum(numCount))
	print ('The average number of words in the files is', sum(numCount)/len(numCount))
	```  
	从下图中可以看到，训练集中评论数量共9870条，分词后的词数量共235343个，每条评论中平均词数量为23个：  
	![words](http://oswrmk9hd.bkt.clouddn.com/686ECA64-1F86-42CF-8E54-55A307FE07D9.png)  
	- 将词通过词向量表达，保存为numpy格式：  
	```
	def make_sequence_matrix(number, comment_list, filename):
    	maxSeqLength = 60
    	matrix = np.zeros((number, maxSeqLength), dtype='int32')
    	commentCounter = 0
    	for comment in comment_list:
        	indexCounter = 0
        	for word in comment:
            	try:
                	matrix[commentCounter][indexCounter] = mod.index2word.index(word)
            	except ValueError:
                	matrix[commentCounter][indexCounter] = len(mod.index2word)-1
            	indexCounter += 1
            	if indexCounter >= maxSeqLength:
                	break
        	commentCounter += 1
    	np.save(filename, matrix)
    ```  
- 基于循环网络(RNN)和长短期记忆网络(LSTM)构建预测模型
NLP数据的一个独特之处是它是时间序列数据。每个单词的出现都依赖于它的前一个单词和后一个单词。由于这种依赖的存在，我们使用循环神经网络来处理这种时间序列数据。  
在 RNN 中，句子中的每个单词都被考虑上了时间步骤。与每个时间步骤相关联的中间状态也被作为一个新的组件，称为隐藏状态向量h(t)，隐藏状态是当前单词向量和前一步的隐藏状态向量的函数。并且这两项之和需要通过激活函数来进行激活。公式如下图所示：  
![rnn1](http://oswrmk9hd.bkt.clouddn.com/1155267-97b8dfddb90a3297.png)  
上面的公式中的2个W表示权重矩阵。其中一个矩阵是和我们的输入x进行相乘。另一个是隐藏的装填向量，用来和前一个时间步骤中的隐藏层输出相乘。W(H)在所有的时间步骤中都是保持一样的，但是矩阵W(x)在每个输入中都是不一样的。这些权重矩阵的大小不但受当前向量的影响，还受前面隐藏层的影响。  
权重的更新，我们采用BPTT算法来进行跟新。在最后的时刻，隐藏层的状态向量被送入一个softmax分类器进行分类。  
长短期记忆网络单元，是另一个RNN中的模块，从抽象的角度看，LSTM 保存了文本中长期的依赖信息，它会判断哪些信息是有用的，哪些是没用的，并且把有用的信息在 LSTM 中进行保存。
LSTM每个单元都将x(t)和h(t-1)作为输入，并且利用这些输入来计算一些中间状态，每个中间状态都会被送入不同的管道，并且这些信息最终会汇集到h(t)，最终输出h(t)。简单来说，前一个LSTM隐藏层的输出是下一个LSTM的输入。具体内容可以参考[Christopher Olah 的博客](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)。
	- 模型超参数变量，批处理大小48，LSTM的单元个数128，分类类别5，迭代次数100000，最大句子长度60，词向量维度300。  
	```
	batchSize = 48
	lstmUnits = 128
	numClasses = 5
	iterations = 100000
	maxSeqLength = 60
	numDimensions = 300
	```  
	- 生成输入数据占位符和标签占位符  
	```
	input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
	labels = tf.placeholder(tf.float32, [batchSize, numClasses])
	```  
	- 使用TensorFlow的嵌入函数构件模型输入词向量，嵌入函数有两个参数，一个是嵌入矩阵（即词向量矩阵），另一个是每个词对应的索引。  
	```
	data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
	data = tf.nn.embedding_lookup(mod.syn0, input_data)
	```  
	- 使用tf.nn.rnn_cell.BasicLSTMCell函数构建LSTM单元，输入的参数表示需要几个LSTM单元；然后设置一个dropout参数，以此来避免一些过拟合；最后，将LSTM cell和三维的数据输入到tf.nn.dynamic_rnn，这个函数的功能是展开整个网络，并且构建一整个RNN模型。  
	```
	lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
	lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
	value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
	```  
	- dynamic RNN函数的第一个输出可以被认为是最后的隐藏状态向量，这个向量将被重新确定维度，然后乘以最后的权重矩阵和一个偏置项来获得最终的输出值。  
    ```
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
	bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
	value = tf.transpose(value, [1, 0, 2])
	last = tf.gather(value, int(value.get_shape()[0])-1)
	predication = tf.matmul(last, weight) + bias
    ```  
    - 定义正确的预测函数和正确率评估参数。  
    ```
    correctPred = tf.equal(tf.argmax(predication, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    ```  
    - 使用一个标准的交叉熵损失函数来作为损失值，并选择Adam优化器，采用默认的学习率。  
    ```
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predication, labels=labels))
	optimizer = tf.train.AdamOptimizer().minimize(loss)
    ```  
    - 使用Tensorboard来可视化前面定义的损失值和正确率。  
    ```
    import datetime
    tf.summary.scalar('Loss', loss)
	tf.summary.scalar('Accuracy', accuracy)
	merged = tf.summary.merge_all()
	logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
	writer = tf.summary.FileWriter(logdir, sess.graph)
    ```  
    - 定义一个TensorFlow会话，开始循环训练。每50次循环将损失值和正确率写入Tensorboard，每10000次循环就保存一次模型断点。  
    ```
    sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
		for i in range(iterations):
    	nextBatch, nextBatchLabels = getTrainBatch()
    	sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
    	if i % 50 == 0:
        	nextBatch_valid, nextBatchLabels_valid = getValidBatch()
        	summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        	writer.add_summary(summary, i)
    	if i % 10000 == 0 and i != 0:
        	save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        	print "saved to %s" % save_path
	writer.close()
    ```  
    - 训练的时间较长，如果使用GPU加速训练的话预计时间是4个小时左右。训练结束后加载模型断点，查看验证集的预测结果，就可以验证模型是否具有较好的泛化能力。  
    ```
    sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint('models'))
	iteration = 10
	for i in range(iteration):
    	next_batch_valid, next_batch_labels_valid = getValidBatch()
    	print ("Accuracy for this batch:", (sess.run(accuracy, {input_data: next_batch_valid,labels: next_batch_labels_valid})))
    ```  
    - 训练模型的损失值和正确率结果如下图所示，可以看到，随着迭代次数的不断增加，损失率不断下降，准确率也不断提升，最终稳定在95%左右。  
    ![loss](http://oswrmk9hd.bkt.clouddn.com/loss.png)  
    ![accuracy](http://oswrmk9hd.bkt.clouddn.com/accuracy.png)  

## 随机森林
    随机森林对于高维数据有很好的分类结果，运行效率和准确率高，实现起来也比较简单。随机森林的生成方法基于以下几步：  
    - 从样本集中用Bootstrap随机选取n个样本；  
    - 从所有属性中随机选取K个属性，选择最佳分割属性作为节点建立CART决策树；  
    - 重复以上两步m次，即建立了m棵CART决策树；  
    - 这m个CART形成随机森林，通过投票表决结果，决定数据属于哪一类；  
    本项目基于scikit-learn库构建随机森林，模型构建主要的任务是调参，我们采用网格搜索进行调参：
    - 当不设制其他参数的时候，模型的精度为**0.937811404318**，袋外分数（反映模型泛化能力）为**0.305775973427**  
    ```
    rf0 = RandomForestClassifier(oob_score=True)
	rf0.fit(X, y)
	print rf0.score(X, y)
	print rf0.oob_score_
    ```  
    - n_estimators是RF最大的决策树个数，max_depth是决策树最大深度，min_samples_split是内部节点再划分所需最小样本数，意思是如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。三个参数同时通过网格搜索进行调参，所得最佳参数结果为：**{'min_samples_split': 6, 'n_estimators': 40, 'max_depth': 23}**  
    ```
    param_test1 = {'n_estimators': range(10, 50, 10), 'max_depth': range(15, 30, 2), 'min_samples_split': range(2, 10, 2)}
	gsearch1 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_test1, scoring='accuracy', cv=5)
	gsearch1.fit(X, y)
	print gsearch1.best_params_
    ```
    - max_features是RF划分时考虑的最大特征数，默认是"None",意味着划分时考虑所有的特征数。通过网格搜索单独对这个参数进行调参，所得最佳参数结果为：**{'max_features': 50}**  
    ```
    param_test2 = {'max_features': range(50, 60, 2)}
	gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=40, min_samples_split=6, max_depth=27), param_grid=param_test2, scoring='accuracy', cv=5)
gsearch2.fit(X, y)
	print gsearch2.best_params_
    ```  
    - 使用之前获得的调参结果，设置随机森林模型的参数，最终获得模型精度**0.93587377745**，袋外分数为**0.36390477948**，可以看到模型的精度已经稳定，但泛化能力经过调参后得到了提高。  
    ```
    rf1 = RandomForestClassifier(n_estimators=40, min_samples_split=6, max_depth=27, max_features=56, oob_score=True)
	rf1.fit(X, y)
	print rf1.score(X, y)
	print rf1.oob_score_
    ```  

## 展望  
- 目前豆瓣电影评论每个电影只显示前500条评论，因此要获取大量数据需要爬取不同电影的评论。本项目采用解析页面的方法爬取数据，是否有别的抓包方法爬取数据还有待研究；  
- 经过不同的模型验证，分类的精度基本上只能达到93%－95%之间，经过研究发现很多评论并不具有感情色彩，或者说评论和评星并不一致，这也会导致模型的精度无法提升到更高；  
- 神经网络和随机森林的模型泛化能力都不高，上述评论和评星不一致也是原因之一，还有的原因是每个电影的评论都有各自电影的特色，很多评论都是各自电影的台词等内容，因此还需要更多高质量的样本数据来提高分类的精度和泛化能力。  

## 和我联系
E-mail: 535848615@qq.com  
GitHub主页: [https://github.com/JohnsonKlose](https://github.com/JohnsonKlose)  
博客园: [http://www.cnblogs.com/KloseJiao/](http://www.cnblogs.com/KloseJiao/)  
喜欢的朋友们可以加个star，也欢迎留言和邮件与我交流！

