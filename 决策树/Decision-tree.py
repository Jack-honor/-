
from math import log
import operator
import pickle


def createDataSet():
    dataSet = [['青年', '否', '否', '一般',    'no'],
            [   '青年', '否', '否', '好',      'no'],
            [   '青年', '是', '否', '好',     'yes'],
            [   '青年', '是', '是', '一般',   'yes'],
            [   '青年', '否', '否', '一般',    'no'],
            [   '中年', '否', '否', '一般',    'no'],
            [   '中年', '否', '否', '好',      'no'],
            [   '中年', '是', '是', '好',     'yes'],
            [   '中年', '否', '是', '非常好', 'yes'],
            [   '中年', '否', '是', '非常好', 'yes'],
            [   '老年', '否', '是', '非常好', 'yes'],
            [   '老年', '否', '是', '好',     'yes'],
            [   '老年', '是', '否', '好',     'yes'],
            [   '老年', '是', '否', '非常好', 'yes'],
            [   '老年', '否', '否', '一般',     'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet,labels


"""
DataSet：给定的数据集
Function：计算给定数据集的香浓熵
return：返回的是计算好的香浓熵
"""
def ShannonEnt(dataSet):
    len_dataSet = len(dataSet)     #样本数据的个数
    label_class = {}               #用来记录每个样本类别的个数
    Ent = 0.0                      #用来存储经验熵
    for item in dataSet:          #循环读入实例
        if item[-1] not in label_class.keys():       #如果存储类别的的字典内没有现在的类别，那么就创建一个以当前类别为key值的元素
            label_class[item[-1]] = 0                 #并将其value值赋值为0
        label_class[item[-1]] += 1                    #如果字典内已经存在此类别，那么将其value加 1，即当前类别的个数加一
    for lable in label_class:                         #从字典内循环获取所有的类别
        P = float(label_class[lable]) / len_dataSet   #计算当前类别占总样本数据的比例，即当前类别出现的概率
        Ent -= P * log(P, 2)        #计算所有类别的香浓熵
    return Ent


"""
dataSet: 给定的数据集
axis: 给定的特征的索引值
value: 对应索引的值
new_dataset: 根据给定的特征划分的新数据
Function：按照给定的特征将数据集分类成新的数据集
"""
def splitDataSet(dataSet, axis, value):
    new_dataset = []
    for item in dataSet:                            #循环读入数据集的每一行数据
        if item[axis] == value:                     #如果是我们需要的特征的数据就将其存放到新的数组中
            templet_set = item[:axis]               #中间变量，用于存放获取的特征变量所在行的其他数据
            templet_set.extend(item[axis+1:])       #a=[1,2,3], b=[4,5,6]   a.extend(b)=[1, 2, 3, 4, 5, 6]
            new_dataset.append(templet_set)         #a=[1,2,3], b=[4,5,6]   a.append(b)=[1, 2, 3, [4,5,6] ]
    return new_dataset


"""
dataSet: 输入的数据集
Function: 选择信息增益最大的特征
return: 返回的是信息增益最大的特征在DataSet中的列索引值
"""
def chooseBestFeature(dataSet):
    len_lables = len(dataSet[0]) - 1     #获取数据集的特征总数。减去的是数据集的类别
    base_Ent = ShannonEnt(dataSet)  #获得总数据集的香农熵
    base_InfoGain = 0.0             #用于记录当前最佳信息增益
    best_lables = -1                #用于记录获得最佳信息增益的特征的索引值
    for i in range(len_lables):    #获取每个特征相应的香农熵
        lable_list = [items[i] for items in dataSet]  #利用列表生成式获取相应特征下的分类，item表示为dataSet的单行数据，item[i]表示对应数据的具体数值
        unique_value = set(lable_list)  #利用集合获得唯一的数据特征，set跟数学中的集合概念类似，里面没有重复的元素
        new_Ent = 0.0                 #用于存放当前子类数据集的经验条件熵
        for value in unique_value:   #获取单个特征值下的对应值例如：青年，中年， 老年
            sub_dataset = splitDataSet(dataSet, i, value)  #按照当前的特征值将数据集进行划分
            prob = len(sub_dataset) / float(len(dataSet))  #获得当前特征的数据占总数据的比例，即概率
            new_Ent += prob * ShannonEnt(sub_dataset)      #获得当前类别的经验条件熵
        info_Gain = base_Ent - new_Ent          #获得当前的信息增益
        #print("第",i,"个特征的信息增益为 ",info_Gain)
        if(info_Gain > base_InfoGain):
            base_InfoGain = info_Gain           #如果遇见更好的信息增益，就将更好的信息增益赋值给best_InfoGain，并且记录下当前信息增益的特征值的索引值
            best_lables = i
    #print("最好的特征索引值为：",best_lables)
    return best_lables

"""
classlist:给定的数据字典
function：如果决策树仍然不能正确分类，那么就采取举手表达的措施选择类数最大的类别
"""
def majorityCnt(classlist):
    classCount = {}                             #创建一个字典，用于存放最大的类别
    for vote in classlist:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""
dataSet:给定的数据集
lables:给定的标签
return:返回的是构造好的决策树
"""
def createTree(dataSet,lables):
    class_list = [example[-1]  for example in dataSet]         #获取数据集的类别
    if class_list.count(class_list[0]) == len(class_list):      #如果数据集都是一种类别的话，那么就返回这个类别
        return class_list[0]
    if len(dataSet[0]) == 1:                                    #如果遍历完所有数据任有数据不能正确分类，那么就采用举手表决的方式，选择数据类最大的一个类别
        return majorityCnt(class_list)
    bestFeat = chooseBestFeature(dataSet)                      #获取最佳特征的索引值
    bestFeatLabel = lables[bestFeat]                           #根据索引值在标签列表里面获得标签的名称
    myTree = {bestFeatLabel:{}}                                #创建一个字典，这个字典用于存放决策树的数据结构
    del(lables[bestFeat])                                      #获得了我们需要的标签之后，就将标签从标签列表里面删除，防止产生死循环
    featValue = [example[bestFeat] for example in dataSet]     #从数据集获得对应标签的所有数据，即最好的特征数据的值
    uniqueValue = set(featValue)                               #利用集合将重复的特征数据的值删除。每个特征的值只留下一个
    for value in uniqueValue:                                  #循环获取特征的值
        subLables = lables[:]                                  #将删除最优特征的标签列表赋值给subLables
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value), subLables)    #递归调用数生成函数，递归生成树的节点
    return myTree                                             #返回特征的树字典结构表达式


"""
inputTree:输入我们构建好的树
featLabels:数据集的标签
testVec:列表，包含树的节点
return:返回的是叶子结点
function:用于测试决策树是否合格
"""
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))                                                        #获取决策树结点
    secondDict = inputTree[firstStr]                                                        #下一个字典
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

"""
function:将决策树持久化
"""
def storeTree(inputTree, filename):
    with open(filename, 'wb') as f:     #以二进制方式写入数据
        pickle.dump(inputTree,f)

"""
function:读取持久化的决策树
"""
def grabTree(filename):
    with open(filename,'rb') as r:       #以二进制方式读取数据
        return pickle.load(r)


if __name__ == "__main__":
    filename = 'C:/Users/lpp/Desktop/lenses.txt'
    with open(filename) as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)