# -*- coding: utf-8 -*-
from math import log
import operator
import pickle
from treeplotter import *

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
        print('infoGain = ', infoGain, 'bestFeature = ', i, 'baseEntropy = ',baseEntropy,'newEntropy = ', newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# myDat, labels = createDataSet()
# print(chooseBestFeatureToSplit(myDat))

def majorityCnt(classList):
    """
    当处理完所有属性后仍有类标签不唯一
    采用多数表决来决定分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    递归构造决策树
    """
    classList = [example[-1] for example in dataSet]
    # 如果只有一个类别就直接返回结果
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    subLabels = labels[:]
    del(subLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# myDat, labels = createDataSet()
# print(createTree(myDat, labels))

def classify(inputTree,featLabels,testVec):
    """
    决策树分类函数
    """
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    """
    存储决策树
    """
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close

def grabTree(filename):
    """
    读取决策树
    """
    fr = open(filename, 'rb')
    return pickle.load(fr)

# myTree = retrieveTree(0)
# createPlot(myTree)

# 示例1: 使用决策树预测隐形眼镜类型

def testLenses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    storeTree(lensesTree, 'trees.txt')

# testLenses()
# createPlot(grabTree('trees.txt'))