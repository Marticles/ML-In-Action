# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 《机器学习实战》 - 第5章 - Logistic回归

# 示例1:采用梯度上升法找到Logistic回归分类器的最佳回归系数

def loadDataSet():
    """
    读取数据集
    """
    dataMat = []
    labelMat = []
    fr = open('TestSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # X0设为1.0
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(z):
    """
    sigmoid函数
    """
    return 1.0 / (1 + np.exp(-z))

def gradAscent(dataMatIn, classLabels):
    """
    梯度上升法
    """
    dataMatrix = np.mat(dataMatIn)
    # 转置为列向量
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001  # 学习率
    maxCycles = 500  # 迭代次数
    weights = np.ones((n,1))  # 权重
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 测试结果
# dataArr, labelMat = loadDataSet()
# print(gradAscent(dataArr,labelMat))

def stoGradAscent0(dataMatrix, classLabels):
    """
    随机梯度下降法
    """
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * dataMatrix[i]* error
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    """
    改进的随机梯度下降法
    """
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    # i和j的不断增大令学习率不断减少，但是不为0
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001
            # 随机抽取样本
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(wei):
    """
    对结果进行可视化
    """
    # weights = wei.getA()  # getA()方法将numpy矩阵转为数组
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, marker = 's')
    ax.scatter(xcord2, ycord2, s = 30,)
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y, c = 'red')
    plt.xlabel(('X1'))
    plt.ylabel(('Y1'))
    plt.show()

# 可视化
# dataArr, labelMat = loadDataSet()
# plotBestFit(stocGradAscent1(dataArr,labelMat))

# 示例2: 从疝气病症预测病马的死亡率

def classifyVector(inX, weights):
    """
    sigmoid分类器
    根据权重与特征来计算sigmoid的值，大于0.5返回1，否则返回0
    """
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    """
    在疝气病马数据集中测试分类效果
    """
    frTrain = open('HorseColicTraining.txt')
    frTest = open('HorseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 在训练集上训练
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    # 在测试集上进行测试
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    """
    调用colicTest()10次并求结果的平均值
    """
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

# 测试结果
# multiTest()