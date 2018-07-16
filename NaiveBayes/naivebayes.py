# -*- coding: utf-8 -*
import numpy as np
import re
import operator
import feedparser

# 《机器学习实战》 - 第4章 - 基于概率论的分类方法：朴素贝叶斯

# 示例1: 使用朴素贝叶斯进行文档分类(正常/侮辱性言论二分类)

def loadDataSet():
    """
    创建数据集
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 为侮辱性言论，0为正常言论
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建词汇表
    """
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符|求并集
    return list(vocabSet)  # vocabSet为去重后词汇表

def setOfWords2Vec(vocabList, inputSet):
    """
    词集模型(set-of-words model)
    用一个向量存放遍历单词是否出现的结果，出现该单词则设为1
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            # print('the word: {} is not in my vocabulary'.format(word))
            pass
    return returnVec  # returnVec = [0,1,0,1...]

def bagOfWords2VecMN(vocabList, inputSet):
    """
    词袋模型(bag-of-words model)
    与词集模型的不同在于每次累加单词出现次数
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            # print('the word: {} is not in my vocabulary'.format(word))
            pass
    return returnVec  # returnVec = [0,1,0,1...]

def _trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯原版
    """
    numTrainDocs = len(trainMatrix)  # 文件数
    numWords = len(trainMatrix[0])  # 单词数
    # 侮辱性文件的出现概率，即trainCategory中所有的1的个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 单词出现次数列表
    p0Num = np.zeros(numWords) # [0,0,0,.....]
    p1Num = np.zeros(numWords) # [0,0,0,.....]
    # 整个数据集单词出现总数
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 如果是侮辱性言论，对侮辱性言论的向量进行加和
            p1Num += trainMatrix[i] #[0,1,1,....] + [0,1,1,....]->[0,2,2,...]
            # 对向量中的所有元素进行求和，也就是计算所有侮辱性言论中出现的单词总数
            p1Denom += sum(trainMatrix[i])
        else:
            # 不是侮辱性言论
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #每个元素做除法，得到1/0(侮辱性/正常)言论分类下每个单词出现概率
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive

def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯修正版，防止乘积为0与分母下溢出
    """
    numTrainDocs = len(trainMatrix)  # 文件数
    numWords = len(trainMatrix[0])  # 单词数
    # 侮辱性文件的出现概率，即trainCategory中所有的1的个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 单词出现次数列表
    # p0Num = np.zeros(numWords) # [0,0,0,.....]
    # p1Num = np.zeros(numWords) # [0,0,0,.....]
    p0Num = np.ones(numWords) # [0,0,0,.....]
    p1Num = np.ones(numWords) # [0,0,0,.....]
    # 整个数据集单词出现总数
    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 如果是侮辱性言论，对侮辱性言论的向量进行加和
            p1Num += trainMatrix[i] #[0,1,1,....] + [0,1,1,....]->[0,2,2,...]
            # 对向量中的所有元素进行求和，也就是计算所有侮辱性言论中出现的单词总数
            p1Denom += sum(trainMatrix[i])
        else:
            # 不是侮辱性言论
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #每个元素做除法，得到1/0(侮辱性/正常)言论分类下每个单词出现概率
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类器
    乘法转加法:
    P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def runNB(testEntry, myVocabList, p0V, p1V, pAb):
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    classified_result = classifyNB(thisDoc, p0V, p1V, pAb)
    if classified_result == 0:
        print(testEntry,'classified as: {}'.format(classified_result),'正常言论')
    else:
        print(testEntry,'classified as: {}'.format(classified_result),'侮辱性言论')

def testingNB():
    """
    测试NB分类器
    """
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    
    # 测试言论1
    testEntry = ['love', 'my', 'dalmation']
    runNB(testEntry,myVocabList,p0V, p1V, pAb)
    
    # 测试言论2
    testEntry = ['stupid', 'garbage']
    runNB(testEntry,myVocabList,p0V, p1V, pAb)

# 测试文档分类
# testingNB()

# 示例2: 使用朴素贝叶斯过滤垃圾邮件

def textParse(bigString):
    """
    切分文本，去掉标点符号并转为小写
    """
    listOfTokens = re.split(r'\W*', bigString)  # 利用正则表达式来切分文本
    return [tok.lower() for tok in listOfTokens if len(tok) > 0]

def spamTest():
    """
    测试垃圾邮件NB分类器
    """
    docList = []
    classList = []
    fullText = []
    # email文件夹中只有25个邮件
    for i in range(1, 26):
        wordList = textParse(open('email/spam/{}.txt'.format(i)).read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/spam/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建词汇表    
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 随机取 10 个邮件用来测试
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(trainMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the errorCount is: ',errorCount)
    print('the testSet length is :',len(testSet))
    print('the error rate is ',format(float(errorCount) / len(testSet)))

# 测试邮件分类
# spamTest()

# 示例3: 使用朴素贝叶斯分类器从个人广告中获取区域倾向

# 由于原书RSS源已失效，改为使用newyork与sfbay的groups(地区组织)源
# 即程序需要将来自newyork与sfbay的groups进行二分类

ny = feedparser.parse('https://newyork.craigslist.org/search/grp?format=rss')
sf  = feedparser.parse('https://sfbay.craigslist.org/search/grp?format=rss')

def calcMostFreq(vocabList,fullText):
    """
    计算TopN高频词
    """
    freqDict = {}
    for token in vocabList:
        # 统计每个词在文本中出现的次数
        freqDict[token] = fullText.count(token)
        # 根据每个词出现的次数从高到底对字典进行排序
    sortedFreq = sorted(freqDict.items(),key = operator.itemgetter(1),reverse = True)
    return sortedFreq[:10]

def stopWords():
    """
    导入停用词，来自http://www.ranks.nl/stopwords
    """
    wordList =  open('stopword/stopword.txt').read()
    listOfTokens = re.split(r'\W*', wordList)
    return listOfTokens

def localWords(feed1,feed0):
    """
    从RSS源获取数据，并测试朴素贝叶斯分类
    """
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)

    # 去除停用词
    stopWordList = stopWords()
    for stopWord in stopWordList:
        if stopWord in vocabList:
            vocabList.remove(stopWord)
    
    """
    # 去除TopN高频词
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    """

    trainingSet = list(range(50))
    testSet = []
    for i in range(5):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    """
    取出前20个热点词
    """
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i]>-6.0:topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0:topNY.append((vocabList[i],p1V[i]))

    sortedSF = sorted(topSF,key = lambda pair:pair[1],reverse = True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF[0:20]:
            print(item[0])

    sortedNY = sorted(topNY,key = lambda pair:pair[1],reverse = True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY[0:20]:
        print(item[0])

def rssTest():
    vocabList,pSF,pNY = localWords(ny,sf)

def topwordsTest():
    getTopWords(ny,sf)

# 测试不同地域关注词
# topwordsTest()