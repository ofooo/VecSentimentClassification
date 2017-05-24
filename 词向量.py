# -*- coding:utf-8 -*-
 
'''
word embedding测试
在GTX960上，18s一轮
经过30轮迭代，训练集准确率为98.41%，测试集准确率为89.03%
Dropout不能用太多，否则信息损失太严重
'''
 
import numpy as np
import pandas as pd
import jieba
import os
import h5py 
def ppath(ff):
	return os.path.join(os.getcwd(),os.path.dirname(__file__),ff)

pos = pd.read_excel(ppath('pos.xls'), header=None)
pos['label'] = 1
neg = pd.read_excel(ppath('neg.xls'), header=None)
neg['label'] = 0
all_ = pos.append(neg, ignore_index=True)
all_['words'] = all_[0].apply(lambda s: list(jieba.cut(s))) #调用结巴分词
 
maxlen = 100 #截断词数
min_count = 5 #出现次数少于该值的词扔掉。这是最简单的降维方法
 
content = []
for i in all_['words']:
	content.extend(i)
 
abc = pd.Series(content).value_counts()		#abc是总词典
abc = abc[abc >= min_count]
abc[:] = range(1, len(abc)+1)	#把词频赋值为排序序号
abc[''] = 0 #添加空字符串用来补全
 
def doc2num(s, maxlen): 
	s = [i for i in s if i in abc.index]
	s = s[:maxlen] + ['']*max(0, maxlen-len(s))
	return list(abc[s])
 
all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen))
 
#手动打乱数据
idx = list(range(len(all_)))
np.random.shuffle(idx)
all_ = all_.loc[idx]
 
#按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1,1)) #调整标签形状
 
 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
 
#建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128)) 
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',  optimizer='adam',  metrics=['accuracy'])
 
#训练模型
batch_size = 128
train_num = 15000
model.fit(x[:train_num], y[:train_num], batch_size = batch_size, nb_epoch=1)
model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)
 
def predict_one(s): #单个句子的预测函数
	s = np.array(doc2num(list(jieba.cut(s)), maxlen))
	s = s.reshape((1, s.shape[0]))
	return model.predict_classes(s, verbose=0)[0][0]
def printone(s):
	print(str(predict_one(s)) + "\t" + s)

printone('卖家他态度特别好，中间虽然出了点问题，但是很快得到了解决，满意，给五分。')
printone('正品，用的很舒服，定位也很，物流相当快，以后还会光顾！')
printone('鼠标挺好用的，底部不会发光应该是节电设计吧！')
printone('第三次购买了，很好用的办公鼠标，手感舒适，价格实惠，好评！')
printone('东西不错，已经使用了。买过几次了。')
printone('刚用觉得有点飘，习惯就好了。')
printone('这鼠标适合手小的人用，除了牌子，感觉一无是处，价钱也不便宜，指向太差了。')
printone('感觉没有照片上面好看啊，重量也有点轻')
printone('收到产品后试用根本不能对中，移动速度一下快一下慢，点不中东西，寄回去检测还说没问题，调了下速率，结果寄回来还是一样的，大家别买了，假货，我用了罗技的不少于10个，从没出过这类问题，对码也对不了，就是路边摊20块一个的也好用')

#保存模型
json_string = model.to_json()  
open(ppath('my_model_architecture.json'),'w').write(json_string)  
model.save_weights(ppath('my_model_weights.h5'))  


while input('输入end结束:') != 'end': pass	#避免误按键关闭程序