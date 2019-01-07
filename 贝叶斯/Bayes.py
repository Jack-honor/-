import numpy as np
from math import log
import re
"""
Function：提供训练数据集，以及训练数据集的标签
"""
def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],				#切分的词条
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]   		#类别标签向量，1代表侮辱性词汇，0代表不是
	return postingList, classVec	    #返回实验样本切分的词条和类别标签向量

"""
Function:创建一个词典，这个词典包含文档内所有词条
dataSet:输入的是训练数据集，即所有的文档
return：返回一个包含所有单词的词典
"""
def creatVocabList(dataSet):
    vocabSet = set([])                               #创建一个空集
    for document in dataSet:                     #从数据集循环读入每一条数据
        vocabSet = vocabSet | set(document)       #采用集合将一条数据的不重复单词放入vocablist，并采用并集操作
                                                  #将曾哥数据集的单词都放入vocabset中
    return list(vocabSet)                        #返回词典，这个词典包含数据集内所有单词，并且将其list序列化


"""
Function:如果数据中的单词在词典中，那么就将词典对应的位置置1
vocabList:输入数据集的词典
inputSet:输入的数据集的一条数据
return:返回的是检测之后的列表
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)      #创建一个全0列表，列表长度跟词典长度一样
    for word in inputSet:              #从单条数据中循环读取所有单词
        if word in vocabList:          #如果单词在词典中，那么就将全0列表returnvec相应的位置置1
            returnVec[vocabList.index(word)] = 1
    return returnVec                #返回的是检测之后的列表

"""
朴素贝叶斯分类器训练函数
Function:计算词条在相应类别下的条件概率
trainMattrix：setOfWords2Vec的返回值，不过是整个训练数据集的返回集而不是单调数据的
trainCategory：loadDataSet()返回的classVec，即数据集对应的类别
return：返回的是两个矩阵一个概率。两个矩阵分别是对应类别下词条的条件概率；概率是侮辱性文档占总文档的概率
"""
def trainNB0(trainMattrix, trainCategory):
    numTrainDocs = len(trainMattrix)       #计算训练数据集总共包含多少文档，为了循环遍历所有文档用的
    numWords = len(trainMattrix[0])        #计算训练数据集有多少个词条，即词典包含多少个词，为了后面建矩阵用的
    pAbusive = sum(trainCategory) / float(numTrainDocs)  #获得类别A，即侮辱性文档占总文档的概率，即先验概率
    p0Num = np.ones(numWords)             #创建两个一维矩阵，大小为词典长度。用于统计每个类别下对应词条的个数
    p1Num = np.ones(numWords)             #便于后续计算概率，即P(x,y,z|C)=P(x|c)P(y|C)P(z|C)，这个例子中只有一个特征
    p0Denom = 2.0                          #这两个数是记录对应类别里面总共有多少个词条的，即分母
    p1Denom = 2.0
    for i in range(numTrainDocs):         #循环遍历数据集所有文档
        if trainCategory[i ] == 1:        #如果文档对应label =1 ，说明是侮辱性文档
            p1Num += trainMattrix[i]      #运用矩阵加法运算，将同一类别下的每个对应词条个数相加
            p1Denom += sum(trainMattrix[i])   #运用数字加法，记录目前总共有多少个词条
        else:
            p0Num += trainMattrix[i]
            p0Denom += sum(trainMattrix[i])
    p1Vect = p1Num / p1Denom        #返回矩阵，矩阵每个元素为对应词条在相应类别下的的条件概率
    p0Vect = p0Num / p0Denom         #返回矩阵，矩阵每个元素为对应词条在相应类别下的的条件概率
    return p0Vect, p1Vect, pAbusive

"""
Function：将给定的数据按照概率的大小非为对应的类别
vec2Classify:待分类的向量
p0Vec:trainNB0返回的三个参数之一
p1Vec:trainNB0返回的三个参数之一
pClass1:trainNB0返回的三个参数之一
return:返回的是分类结果
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)      #因为采用了对数运算，对数乘可以拆分为多个对数相加
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p0 > p1:
        return 0
    else:
        return 1



def test():
    dataSet, Labels = loadDataSet()
    vocabList = creatVocabList(dataSet)
    trainMat = []
    for item in dataSet:
        returnVec = setOfWords2Vec(vocabList, item)
        trainMat.append(returnVec)
    p0Vect, p1Vect, pAbusive = trainNB0(trainMat, Labels)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry, 'classifed as：', classifyNB(thisDoc,p0Vect, p1Vect, pAbusive))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry, 'classifed as：', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))

"""
Function：切分文本数据
bigstring：需要切分的数据
"""
def textParse(bigstring):
    listOfTokens = re.split(r'\W*', bigstring)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]; classsList=[]; fullText=[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.text' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classsList.append(1)
        wordList = textParse(open('email/ham/%d.text' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classsList.append(0)
    vocabList = creatVocabList(docList)
    trainingSet = range(50); testSet=[]
    for i in range(10):
        randIndex = int(np.rangdom.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classsList[docIndex])
    p0V, p1V,PSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, PSpam) != classsList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))