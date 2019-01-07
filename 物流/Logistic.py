# -*- coding:UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random
"""
Function：打开文本文档，并将里面的数据读取出来
"""
def loadDataSet():
    dataMat = []  # 创建数据列表
    labelMat = []  # 创建标签列表
    fr = open('C:/Users/lpp/Desktop/testSet.txt')  # 打开文件
    for line in fr.readlines():  # 逐行读取
        lineArr = line.strip().split()  # 去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(int(lineArr[2]))  # 添加标签
    fr.close()  # 关闭文件
    return dataMat, labelMat  # 返回

"""
Function：Sigmoid函数的代码实现
"""
def Sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

"""
Function：梯度上升算法
dataMatIng：输入的数据集
LabelMatIn：数据集的标签
"""
def gradAscent(dataMatIn, labelMatIn):
    dataMatrix = np.mat(dataMatIn)             #将数组转换为numpy矩阵
    labelMat = np.mat(labelMatIn).transpose()   #将数组转换为numpy矩阵，并将其转置
    m, n =np.shape(dataMatrix)                 #获得矩阵的行列数
    alpha = 0.001                              #移动步长
    numb = 500                                 #迭代次数为500次
    weights = np.ones((n, 1))                  #初始化权重为 1，由于是矩阵之间的运算，因此需要创建一个n*1维的全1矩阵
    for i in range(numb):
        oldWeight = Sigmoid(dataMatrix * weights)
        error = (labelMat - oldWeight)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=250):
    """
    改进的随机梯度上升算法
    :param dataMatrix: 输入的数据集
    :param classLabels: 数据集的标签
    :param numTter: 循环迭代次数，默认为250次
    :return: 最优化的系数
    """
    m, n = np.shape(dataMatrix)                               # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                       # 参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01                    # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            h = Sigmoid(sum(dataMatrix[randIndex] * weights))   # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            del (dataIndex[randIndex])                         # 删除已经使用的样本
    return weights

def classifyVector(inX, weights):
    """
    分类函数
    :param inX:输入的数据集
    :param weights:经过训练得到的最优化系数
    :return:返回的是类别
    """
    prob = Sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('C:/Users/lpp/Desktop/horseColicTraining.txt')  #打开训练数据集
    frTest = open('C:/Users/lpp/Desktop/horseColicTest.txt')        #打开测试数据集
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeights = stocGradAscent1(np.array(trainingSet),trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainingWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("The error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("After %d iterations the average error rate is: %f"%(numTests, errorSum / float(numTests)))



def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='green',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='blue')
    plt.title("dataSet")
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == "__main__":
    multiTest()
