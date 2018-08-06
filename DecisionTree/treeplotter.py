# -*- coding: utf-8 -*-
import pylab
from matplotlib import pyplot as plt

# 决策树的可视化

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 解决中文乱码
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    nodeTxt:要显示的文本
    centerPt:文本的中心点，箭头所在的点
    parentPt:指向文本的点
    """
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',\
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

"""
def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode(U"决策节点",(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(U"叶子节点",(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
"""

def getNumLeafs(myTree):
    """
    获取叶子节点数目
    """
    numLeafs = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    # 遍历得到的secondDict
    for key in secondDict.keys():
        # 如果secondDict[key]为一个字典，即决策树结点
        if type(secondDict[key]).__name__ == 'dict':
            # 则递归的计算secondDict中的叶子结点数，并加到numLeafs上
            numLeafs += getNumLeafs(secondDict[key])
        # 如果secondDict[key]为叶子结点
        else:
            # 将叶子结点数加1
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    获取决策树层数
    """
    maxDepth = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 如果secondDict[key]为一个字典
        if type(secondDict[key]).__name__ == 'dict':
            # 则当前树的深度等于1加上secondDict的深度，只有当前点为决策树点深度才会加1
            thisDepth = 1 + getTreeDepth(secondDict[key])
            # 如果secondDict[key]为叶子结点
        else:
            # 将当前树的深度设为1
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt,parentPt,txtString):
    """
    绘制中间文本
    """
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    cntrPt = (plotTree.xOff + (1.0 +float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt, leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff +1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=True,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = - 0.5 / plotTree.totalW
    print(plotTree.xOff)
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

def retrieveTree(i):
    listOfTree = [{'no surfacing':{ 0:'no',1:{'flippers': \
                       {0:'no',1:'yes'}}}},
                   {'no surfacing':{ 0:'no',1:{'flippers': \
                    {0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
                  ]
    return listOfTree[i]