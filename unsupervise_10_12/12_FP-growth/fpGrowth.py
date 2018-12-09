#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'


# 创建一个类保存树的每个节点
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # 存放节点名字
        self.count = numOccur  # 计数
        self.nodeLink = None   # 用于链接相似的元素项
        self.parent = parentNode  # 指向当前节点的父节点
        self.children = {}        # 存放节点的子节点

    def inc(self, numOccure):  # 对count变量增加指定值
        self.count += numOccure

    def disp(self, ind=1):     # 将树以文本形式显示
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)


def createTree(dataSet, minSup=1):
    '''

    :param dataSet:数据集
    :param minSup: 最小支持度
    :return:
    '''
    headerTable = {}
    # go over dataSet twice
    for trans in dataSet.keys():  # first pass counts frequency of occurance
        # print(trans)
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            # print(k, headerTable[k])
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    print('freqItemSet:', freqItemSet)
    # 如果没有元素项满足要求，则退出
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)

    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        # print('localD:', localD)
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # print(orderedItems)
            updateTree(orderedItems, retTree, headerTable, count)  # populate tree with ordered freq itemset
    return retTree, headerTable  # return tree and header table


def updateTree(items, inTree, headerTable, count):
    '''
    让FP树生长
    :param items: 项集（已按出现次数从大到小排好序）
    :param inTree: FP树
    :param headerTable: 头指针表
    :param count: 计数
    :return:
    '''
    if items[0] in inTree.children:   # check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count)   # incrament count
    else:   # add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:  # update header table
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):   # this version does not use recursion
    while (nodeToTest.nodeLink != None):    # Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def ascendTree(leafNode, prefixPath):
    '''
    迭代上溯整棵树
    :param leafNode:
    :param prefixPath:
    :return:
    '''
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    '''

    :param basePat: 某一个频繁项，eg 'x'
    :param treeNode: 头指针表的首个链接点 eg myHeaderTab['x'][1]
    :return: 所有的前缀路径，条件模式基字典
    '''
    condPats = {}  # 条件模式基字典
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath)>1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]

    for basePat in bigL:  # start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # print('finalFrequent Item: ', newFreqSet)    #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print('condPattBases :', basePat, condPattBases)
        # 2. construct cond FP-tree from cond. pattern base
        # 从条件模式基来构建FP树
        myCondTree, myHead = createTree(condPattBases, minSup)
        # print('head from conditional tree: ', myHead)
        if myHead != None:  # 3. mine cond. FP-tree
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
'''
发现Twitter源中的共现词：
（1）收集数据： 使用python-twitter模块来访问推文
（2）准备数据：编写一个函数去掉UR、去掉标点、转换成小写，并从字符串中建立一个单词集合
（3）分析数据：在python提示符下查看准备好的数据，确保它的正确性
（4）训练算法：使用FP-Growth算法
（6）使用算法：可以用于情感分析或者查询推荐领域
'''
import twitter
from time import sleep
import re


def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages


def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList


if __name__ == '__main__':
    # rootNode = treeNode('pyramid', 9, None)
    # rootNode.children['eye'] = treeNode('eye', 13, None)
    # rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    # rootNode.disp()
    simpDat = loadSimpDat()
    # print('simpDat:', simpDat)
    initSet = createInitSet(simpDat)
    print('initSet:', initSet)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    # myFPtree.disp()

    # print(findPrefixPath('x', myHeaderTab['x'][1]))

    # freqItems = []
    # mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    # print(freqItems)

    parseDat = [line.split() for line in open('kosarak.dat').readlines()]
    initSet = createInitSet(parseDat)
    myFPtree, myHeaderTab = createTree(initSet, 100000)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
    print(myFreqList)