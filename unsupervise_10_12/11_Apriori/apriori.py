#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5], [1, 2, 4, 5], [2, 3, 4, 5], [2, 4, 5]]


def createC1(dataSet):
    C1 = []  # 是大小为1的所有候选项集的集合
    for tranction in dataSet:
        for item in tranction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))  # 对C1中每个项构建一个不变的集合


def scanD(D, Ck, minSupport):
    '''

    :param D:数据集
    :param Ck: 候选项集列表
    :param minSupport: 最小支持度
    :return: 包含支持度的字典supportData，满足最小支持度的项集列表
    '''
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
            supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):  # create Ck
    '''
    创建候选项集Ck
    :param Lk: 频繁项集列表Lk
    :param k: 项集元素个数k
    :return:
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2])>0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)  # scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    '''

    :param L:频繁项集列表
    :param supportData: 包含那些频繁项集支持数据的字典
    :param minConf: 最小可信度阈值
    :return: 包含可信度的规则列表
    '''
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        # print('li:', L[i])
        for freqSet in L[i]:
            # print('freqset:', freqSet)
            H1 = [frozenset([item]) for item in freqSet]
            # print('h1:', H1)
            if (i > 1):   # 如果集合中的元素个数超过2，对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:   # 集合中的元素个数等于2，计算可信度值
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    '''
    对规则进行评估
    :param freqSet: 频繁项集中的某个项集
    :param H: 包含单个元素的集合、出现在规则右部的元素列表
    :param supportData: 支持度数据
    :param brl: 存储通过检查的规则集
    :param minConf: 最小可信度
    :return: 满足最小可信度要求的规则列表
    '''
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
    生成候选规则集合
    :param freqSet: 频繁项集
    :param H: 出现在规则右部的元素列表
    :param supportData:
    :param brl:
    :param minConf:
    :return:
    '''
    # print(H)
    m = len(H[0])
    # print('m:', m)
    if (len(freqSet) > (m + 1)): #try further merging
        print('****')
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        # print('Hmp1:', Hmp1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # print('Hmp2:', Hmp1)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


from time import sleep
from votesmart import votesmart  # 此模块需要单独下载

votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'  # 这里需要改换成自己的API key
# 收集美国国会议案中action ID的函数
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])  # 得到了议案的ID
        try:
            billDetail = votesmart.votes.getBill(billNum)  # 得到一个billDetail对象
            for action in billDetail.actions:  # 遍历议案中的所有行为
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:  # API调用时发生错误
            print("problem getting bill %d" % billNum)
        sleep(1)  # 礼貌访问网站而做出些延迟，避免过度访问
    return actionIdList, billTitleList


# 基于投票数据的事物列表填充函数
def getTransList(actionIdList, billTitleList):
    itemMeaning = ['Republican', 'Democratic'] # 创建一个含义列表
    for billTitle in billTitleList: # 遍历所有的议案
        itemMeaning.append('%s -- Nay' % billTitle) # 在议案标题后面添加Nay(反对)
        itemMeaning.append('%s -- Yea' % billTitle) # 在议案标题后添加Yea(同意)
    transDict = {} # 用于加入元素项
    voteCount = 2
    for actionId in actionIdList: # 遍历getActionIds()返回的每一个actionId
        sleep(3) # 延迟访问，防止过于频繁的API调用
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId) # 获得某个特定的actionId的所有投票信息
            for vote in voteList: # 遍历投票信息
                if not transDict.has_key(vote.candidateName):  # 如果没有该政客的名字
                    transDict[vote.candidateName] = [] # 用该政客的名字作为键来填充transDict
                    if vote.officeParties == 'Democratic':  # 获取该政客的政党信息
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning  # 返回事物字典和元素项含义列表


if __name__ == '__main__':
    dataSet = loadDataSet()
    # print(dataSet)
    # C1 = createC1(dataSet)
    # print('C1:', C1)
    # D = list(map(set, dataSet))
    # print('D:', D)
    # L1, supportData0 = scanD(D, C1, 0.5)
    # print('L1:', L1, 'supportData0:', supportData0)

    # L, supportData = apriori(dataSet)
    # print('L:', L)
    # print('supportData:', supportData)
    # print(aprioriGen(L[1], 3))

    # L, suppData = apriori(dataSet, minSupport=0.4)
    # print('L:', L)
    # print('supportData:', suppData)
    # rules = generateRules(L, suppData, minConf=0.7)
    # print(rules)

    # rules = generateRules(L, suppData, minConf=0.5)
    # print(rules)

    # actionIdList, billTitleList = getActionIds()
    # transDict, itemMeaning = getTransList(actionIdList, billTitleList)
    # dataSet = [transDict[key] for key in transDict.keys()]
    # L, suppData = apriori(dataSet, 0.5)  # 得到频繁项集
    # rules = generateRules(L, suppData, 0.99)

    mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
    # print(mushDataSet[0])
    L, supportData = apriori(mushDataSet, minSupport=0.3)
    for item in L[1]:
        # s.intersection(t)
        # s & t
        # 返回一个新的set包含s和t中的公共元素

        if item.intersection('2'):
            print(item)
    for item in L[3]:
        if item.intersection('2'):
            print(item)