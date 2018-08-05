# -*- coding: utf-8 -*-
from math import log

# 《机器学习实战》 - 第3章 - 决策树

def calcShannonEnt(dataSet):
    """
    计算香农熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 每行数据的最后一个是分类标签，标签也是key
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    """
    创建数据集
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def splitDataSet(dataSet, index, value):
    """
    划分数据集，取出index对应的值为value的数据
    dataSet: 待划分的数据集
    index:   划分数据集的特征
    value:   需要返回的特征的值
    """
    retDataSet = []
    for featVec in dataSet: 
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            retDataSet.append(reducedFeatVec)
    # 返回index列为value的数据集(去除index列)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # set去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历某一列的value集合，计算该列的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet)/float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益是熵的减少或者是数据无序度的减少
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature