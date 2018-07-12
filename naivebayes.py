# -*- coding: utf-8 -*
import numpy as np

def loadDataSet():
    '''
    创建数据集
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 为侮辱性言论，0为正常言论
    return postingList, classVec   # postingList单词列表,classVec所属类别


def createVocabList(dataSet):
    '''
    创建词汇表
    '''
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符|求并集
    return list(vocabSet)  # vocabSet为去重后词汇表

def setOfWords2Vec(vocabList, inputSet):
    """
    用一个向量存放遍历单词是否出现的结果，出现该单词则设为1
    """
    # 创建一个元素全为0的向量
    returnVec = [0] * len(vocabList)  # [0,0......]
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: {} is not in my vocabulary'.format(word))
    return returnVec  # returnVec = [0,1,0,1...]

listposts,listclasses= loadDataSet()
vocablist = createVocabList(listposts)
wordvoc = setOfWords2Vec(vocablist,listposts[1])
print(vocablist)
print(wordvoc)
