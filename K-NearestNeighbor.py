#-*-cording:UTF-8-*-
import numpy as np

#将数据从文本读入矩阵中
def ReadFile(filename):
    fr = open(filename)
    ReadLines = fr.readlines()
    Len_File = len(ReadLines)
    Array_return = np.zeros((Len_File,3))
    ClassLables = []
    index = 0
    for line in ReadLines:
        line = line.strip()
        line = line.split('\t')
        Array_return[index, : ] = line[0:3]
        if line[-1] == 'didntLike':
            ClassLables.append(1)
        elif line[-1] == 'smallDoses':
            ClassLables.append(2)
        elif line[-1] == 'largeDoses':
            ClassLables.append(3)
        index += 1
    return Array_return, ClassLables

#归一化处理
def AutoNorm(DataSet):
    array_len = DataSet.shape[0]
    min_value = DataSet.min(axis = 0)
    max_value = DataSet.max(axis = 0)
    value_range = max_value - min_value
    #newArray = np.zeros(np.shape(DataSet))
    newArray = DataSet - np.tile(min_value,(array_len,1))
    newArray = newArray / np.tile(value_range,(array_len,1))
    return newArray,value_range, min_value

#K近邻处理算法
def Classify(Data, DataSet, Lables, k):
    LenData = DataSet.shape[0]
    ArryMat = np.tile(Data, (LenData, 1)) - DataSet
    SqlMat = ArryMat ** 2
    SumMat = SqlMat.sum(axis=1)
    DistanceMat = SumMat ** 0.5
    SortDistance = DistanceMat.argsort()
    ClassCount = {}
    for i in range(k):
        votoLables = Lables[SortDistance[i]]
        ClassCount[votoLables] = ClassCount.get(votoLables,0) + 1
    SortClassCount = sorted(ClassCount.items(), reverse=True)
    return SortClassCount[0][0]




if __name__ == "__main__":
    filename = "C:/Users/lpp/Desktop/datingTestSet.txt"
    DataSet, Lables = ReadFile(filename)
    Array, value_range, min_value = AutoNorm(DataSet)
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    data = np.array([precentTats, ffMiles, iceCream])
    ClassCount = Classify(data, Array, Lables, 2)
    if ClassCount == 1:
        print("不喜欢")
    elif ClassCount == 2:
        print("感兴趣")
    elif ClassCount == 3:
        print("非常喜欢")

