# -*- coding: utf-8 -*-
import numpy as np

# 《机器学习实战》 - 第11章 - Apriori

def loadDataSet():
    """
    创建Hole Foods数据集
    """
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    """
    构建大小为1的所有候选项集(不重复)
    """
    C1 = []
    # 遍历每条记录
    for transaction in dataSet:
        # 对于每一条记录，遍历记录中的每一项
        for item in transaction:
            # 如果没在C1中出现过，添加至C1中
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # frozenset是不可改变的
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
    """
    扫描C1生成满足最小支持度的集合L1
    D:输入数据集
    Ck:候选集列表
    minSupport:最小支持度
    """
    ssCnt = {} # 出现次数
    numItems = float(len(D)) # 输入数据集大小
    retList = [] # 频繁项
    supportData = {} # 频繁项集支持度
    # 遍历输入数据集
    for tid in D:
        # 遍历C1中的所有候选集
        for can in Ck:
            # 如果C1中的候选集在输入数据集中，增加计数
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # ssCnt的key就是C1中的候选集
    # 遍历候选集，计算支持度
    for key in ssCnt.keys():
        support = float(ssCnt[key])/float(numItems)
        # 满足最小支持度要求则添加
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support
    return retList,supportData

# 组织完整的Apriori算法

def aprioriGen(Lk, k):
    '''
    生成候选集Ck
    Lk:输入频繁项集列表
    k：项集元素个数
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    '''
    生成满足最小支持度的候选集
    '''
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def testApriori():
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1, suppData0 = scanD(D, C1, 0.2)
    L, suppData = apriori(dataSet)
    print("dataSet:",dataSet,"\n")
    print("C1",C1,"\n")
    print("L1:",L1,"\n")
    print("L:",L,"\n")

# 从频繁项挖掘关联规则

def generateRules(L,supportData,minConf = 0.7):
    """
    生成关联规则
    """
    bigRuleList = []
    # 只获取两个或者更多的频繁项集合
    for i in range(1,len(L)):
        for freqSet in L[i]:
            # 对每个频繁项集，创建只包含单个元素的列表
            H1 = [frozenset([item]) for item in freqSet]
            # 如果频繁项集元素超过2则合并
            if i > 1:
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            # 频繁项集元素为2，直接计算可信度
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

def calcConf(freqSet,H,supportData,bigRuleList,minConf):
    """
    计算可信度
    """
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,"-->",conseq,"   conf:",conf)
            bigRuleList.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,bigRuleList,minConf):
    """
    进一步合并关联规则
    """
    m = len(H[0])
    if len(freqSet) > (m+1):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,bigRuleList,minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, bigRuleList, minConf)

def :
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet,minSupport=0.5)
    rules = generateRules(L, suppData, minConf = 0.5)
    print(rules)

# 示例1: 发现国会投票中的模式
# 由于API与votesmart的原因，跳过

def getActionIds():
    """
    获取投票数据的actionId
    """
    actionIdList = []
    billTitleList = []
    votesmart.apikey = 'get your api key first'
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) # api call
            for action in billDetail.actions:
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print ('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print ("problem getting bill %d" % billNum)
    return actionIdList, billTitleList

def getTransList(actionIdList,billTitleList):
    """
    基于投票数据的事物列表填充函数
    """
    itemMeaning = ['Republican','Democratic']
    for billTitle in billTitleList:
        itemMeaning.append('%s -- Nay'%billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}
    voteCount = 2
    for actionId in actionIdList:
        print ('getting votes for actionId: %d' %actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties =='Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties=='Republican':
                        transDict[vote.candidateName].append(0)
                    if vote.action=='Nay':
                        transDict[vote.candidateName].append(voteCount)
                    elif vote.action=='Yea':
                        transDict[vote.candidateName].append(voteCount+1)
        except:
            print("problem getting actionId:%d" %actionId)
        voteCount+=2
    return transDict,itemMeaning

# 示例2: 发现毒蘑菇的相似特征

# 第一个特征表示有毒或者可食用。如果某样本有毒，则值为2，如果可食用，则值为1
# 第二个特征是蘑菇伞的形状，有六种可能的值，分别用整数3-8来表示
mushDatSet = [line.split() for line in  open('mushroom.dat').readlines()]
L, suppData = apriori(mushDatSet,minSupport=0.3)

def testMush():
    for item in L[3]:
        if item.intersection('2'):print(item)

# testApriori()
# testRules()
# testMush()
