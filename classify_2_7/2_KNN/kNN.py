#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

# from numpy import *
import numpy as np
import operator
from os import listdir

# 创建数据集和标签
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''

对未知类别属性的数据集中的每个点执行以下操作：
1 计算已知类别数据集中的点与当前点的距离
2 按照距离大小递增排序
3 选取与当前距离最小的k个点
4 确定前k个点所在类别出现的频率
5 返回前k个点出现频率最高的类别作为当前点的类别
'''


def classify0(dataset, labels, inX, k):
    '''

    :param dataset: 训练数据向量
    :param labels: 训练数据的标签标签向量
    :param inX: 用于分类的输入向量
    :param k: 选择最近邻的数目
    :return: 分类标签
    '''
    datasetSize = dataset.shape[0]
    # 距离计算
    diffMat = np.tile(inX, (datasetSize, 1))-dataset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  # axis = 1每一行相加，axis = 0每一列相加
    distances = sqDistances**0.5
    sortedDistances = distances.argsort()  # 将元素从小到大排列，并输出下标的index

    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistances[i]]
        classCount[votelabel] = classCount.get(votelabel, 0)+1  # 存入当前label以及对应的类别值，d.get(k, v)意思是如果k在d中，则返回d[k]，否则返回v
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):

    '''
    将文件中的数据集转成矩阵形式
    :param filename: 文件名
    :return: 数据矩阵和标签向量
    '''

    fr = open(filename)
    oneArrays = fr.readlines()
    m = len(oneArrays)
    dataSet = np.zeros((m, 3))
    labels = []
    index = 0
    for line in oneArrays:
        line = line.strip().split('\t')
        dataSet[index, :] = line[0:3]
        labels.append(int(line[3]))
        index += 1
    return dataSet, labels

def autoNorm(dataSet):
    '''
    归一化特征值
    newVals = (oldVals-minvals)/(maxvals-minvals)
    :param dataSet:训练矩阵
    :return: 归一化的训练矩阵
    '''
    m = dataSet.shape[0]
    minvals = dataSet.min(0)
    max = dataSet.max(0)
    norm_datavec = np.zeros(dataSet.shape)
    ranges = max - minvals
    norm_datavec = dataSet - np.tile(minvals, (m, 1))
    norm_datavec = norm_datavec/np.tile(ranges, (m, 1))
    return norm_datavec, ranges, minvals


def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print
        "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print
    "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print
    errorCount


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print
        "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print
    "\nthe total number of errors is: %d" % errorCount
    print
    "\nthe total error rate is: %f" % (errorCount / float(mTest))


if __name__ == '__main__':
    # sortedClassCount = {'2': 1, '3': 4}
    # sortedClassCounts = sorted(sortedClassCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedClassCounts)

    # group, labels = createDataSet()
    # print(classify0(group, labels, [0.1, 0], 3))

    datingDateMat, datlingLabels = file2matrix('datingTestSet2.txt')
    norm_datingDataMat, ranges, minvals = autoNorm(datingDateMat)
    print(norm_datingDataMat[:10], ranges, minvals)