# -*- coding: utf-8 -*-
import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir

# 《机器学习实战》 - 第2章 - k-近邻算法

def classify0(inX, dataSet, labels, k):
    """
    利用k-近邻算法实现分类，采用欧式距离
    inX: 用于分类的输入向量
    dataSet: 训练集
    labels: 标签向量
    k: 选择最近邻数目
    """
    dataSetSize = dataSet.shape[0]
    # 将输入向量按行复制，与训练集相减得到差值
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 各个差值分别平方
    sqDiffMat = diffMat ** 2
    # 按行对结果求和
    sqDistances = sqDiffMat.sum(axis = 1)
    # 再开方即可得到距离
    distances = sqDistances ** 0.5
    # argsort()方法将向量中每个元素进行排序，结果是元素的索引形成的向量
    # 如argsort([1,3,2]) -> ([0,2,1])
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i  in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中将该类型+1
        # 字典的get()方法
        # 如：list.get(k,d)get相当于一条if...else...语句
        # 参数k在字典中，字典将返回list[k]
        # 如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 字典的 items()方法，以列表返回可遍历的(key，value)元组
    # sorted()中的第2个参数key=operator.itemgetter(1)表示按第2个元素进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回第0个tuple的第0个参数，由于是逆序排序所以返回的是出现次数最多的类型
    return sortedClassCount[0][0]

# 示例1: 使用k-近邻算法改进约会网站的配对效果

def file2matrix(filename):
    """
    导入约会网站数据
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

"""
# 对数据进行可视化
datingDataMat, datingLabels = file2matrix('datingTestSET.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels),edgecolors= 'black')
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels),edgecolors= 'black')
plt.show()
"""

def autoNorm(dataSet):
    """
    数据归一化(Min-Max Scaling)
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def dataingClassTest():
    """
    测试kNN优化后约会网站的配对效果
    """
    hoRatio = 0.10  # 测试集比例
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 数据归一化
    m = normMat.shape[0]  # 训练样本量
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print("the total error count is:",errorCount)

# 测试kNN在约会网站的配对的错误率
# dataingClassTest()

def classifyPerson():
    """
    约会网站预测
    """
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

# 测试约会网站预测
# classifyPerson()

# 示例2: 使用k-近邻算法进行手写数字识别

def img2vector(filename):
    """
    将32*32的图像数据转换为1*1024的向量
    循环读出文件的前32行，并将每行的前32个值存储在1*1024的numpy数组中
    """
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    """
    kNN手写数字识别测试
    """
    # 导入训练集
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 导入测试集
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total correct rate is: %f" % (1-(errorCount / float(mTest))))

# 测试手写数字识别正确率
# handwritingClassTest()