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
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = labelMat[i] - h
        weights = weights + alpha * dataMatrix[i] * error
        weights=np.mat(weights).reshape((3,1))
    return weights

def plotBestFit(wei):
    """
    对结果进行可视化
    """
    weights = wei.getA()  # getA()方法将numpy矩阵转为数组
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
dataArr, labelMat = loadDataSet()
plotBestFit(stoGradAscent0(dataArr,labelMat))




# 示例2: 