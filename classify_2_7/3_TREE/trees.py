#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'


# ID3

import numpy as np
import operator
import math
from collections import Counter
import treePlotter

def calcShannonEnt(dataSet):
    '''
    计算香农熵
    :param dataSet: 带标签的数据集
    :return: 香农熵
    '''
    classLabel = [example[-1] for example in dataSet]
    num_class_every = Counter(classLabel)
    # classLabelUni = set(classLabel)
    # num_class = len(classLabelUni)
    numEntries = len(dataSet)
    shannon = 0.0
    for k in num_class_every.keys():
        prob = num_class_every[k] / float(numEntries)
        shannon -= prob * math.log(prob, 2)
    return shannon


def createDataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [1, 0, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def splitDataset(dataSet, axis, value):
    '''
    按照给定的特征划分数据集
    :param dataSet: 代划分的数据集
    :param axis: 划分数据集的特征
    :param value: 特征的某个值
    :return: 划分后的数据集
    '''

    retData = []
    for feaVec in dataSet:
        if feaVec[axis] == value:
            reducedFecVec = feaVec[:axis]
            reducedFecVec.extend(feaVec[axis+1:])
            retData.append(reducedFecVec)
    return retData


def chooseBestFeatureToSplit(dataset):
    '''
    选择最好的数据集划分方式
    :param dataset: 带标签的数据集
    :return: 最好的划分数据集的特征
    '''
    m = len(dataset)
    num_feature = len(dataset[0]) - 1
    shannon = calcShannonEnt(dataset)
    bestFeature = -1
    bestInfogain = 0.0

    for i in range(num_feature):
        attriVal = [example[i] for example in dataset]
        attriValUni = set(attriVal)
        info = 0.0
        for j in range(len(attriVal)):
            resDataSet = splitDataset(dataset, i, attriVal[j])
            prob = len(resDataSet) / m
            info -= prob * calcShannonEnt(resDataSet)
        infogain = shannon-info
        if infogain>bestInfogain:
            bestInfogain = infogain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    多数表决方法
    :param classList:标签列表
    :return: 标签中类别最多的类别
    '''
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataset, labels):
    '''
    构造决策树
    :param dataset:带标签的数据集
    :param labels: 属性列表
    :return: 树
    '''
    classList = [example[-1] for example in dataset]
    # 类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类别
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]

    featValuesUni = set(featValues)
    for value in featValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataset(dataset, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    '''
    进行预测
    :param inputTree:构造好的ID3决策树
    :param featLabels: 属性列表
    :param testVec: 测试数据
    :return: 测试数据所属的类别
    '''
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # print(firstStr)
    # print(featLabels)
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                # print(secondDict[key])
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    '''
    决策树的存储
    :param inputTree:要存储的决策树
    :param filename: where
    :return:
    '''
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    '''
    提取决策树
    :param filename:存放决策树的文件名
    :return: 决策树
    '''
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == "__main__":
    myDat, labels = createDataset()
    label1 = labels[:]
    # print(calcShannonEnt(myDat))
    # print(chooseBestFeatureToSplit(myDat))
    myTree = createTree(myDat, labels)
    print(myTree)
    print(classify(myTree, label1, [0, 0]))
    print(classify(myTree, label1, [1, 1]))

    # myDat = []
    # labels = []
    # fr = open('lenses.txt')
    # for line in fr:
    #     line = line.strip().split('\t')
    #     myDat.append(line)
    #     labels.append(line[-1])
    # lensesTree = createTree(myDat, labels)
    # print(lensesTree)
    # storeTree(lensesTree, 'classifierStorage.pkl')
    # print(grabTree('classifierStorage.pkl'))

    # treePlotter.createPlot(lensesTree)
