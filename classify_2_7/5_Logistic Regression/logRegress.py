#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import numpy as np


def createDataSet():
    dataMat = []
    labelMat = []
    fr = open('../testSet.txt')
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append([0, float(line[0]), float(line[1])])
        labelMat.append(int(line[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return np.longfloat(1/(1+np.exp(-inX)))


def gradAscent(dataMatIn, classLabel):
    '''
    批量梯度上升
    :param dataMatIn:数据集不带标签
    :param classLabel: 标签
    :return: 回归系数
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabel).transpose()
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))
    alpha = 0.001
    maxCycle = 500
    for i in range(maxCycle):
        h = sigmoid(dataMatrix*weights)
        error = labelMat - h
        weights = weights + alpha *dataMatrix.transpose()*error
    return weights


def plotBestFit(weights):
    # print(np.shape(weights))
    dataMat, labelMat = createDataSet()
    m, n = np.shape(dataMat)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if labelMat[i] == 0:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1])
            ycord2.append(dataMat[i][2])
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=10, c='green', marker='o')
    x = np.arange(-3.0, 3.0, 0.1)
    # print(x)
    # z = weights[1]*x
    # print(np.shape(z))
    y = (-weights[0]-weights[1]*x) / weights[2]
    # print(y)
    ax.plot(x, y.T)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabel):
    '''
    随机梯度上升算法
    :param dataMatrix:
    :param classLabel:
    :return:
    '''
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones((n, 1))
    for i in range(m):
        # print(np.mat(dataMatrix[i]))
        h = sigmoid(np.mat(dataMatrix[i])*weights)
        error = classLabel[i] - h
        weights = weights + alpha * np.mat(dataMatrix[i]).T * error
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0, len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(np.mat(dataMatrix[randIndex])*weights)
            error = classLabels[randIndex] - h
            weights = weights + alpha * np.mat(dataMatrix[randIndex]).T * error
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(np.mat(inX) * weights)
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    # dataMat, classLabel = createDataSet()
    # weights = gradAscent(dataMat, classLabel)
    # print(weights)
    # # weights = stocGradAscent0(dataMat, classLabel)
    # # weights = stocGradAscent1(dataMat, classLabel)
    # plotBestFit(weights)
    # print(classifyVector([0, -1.076637, -3.181888], weights))

    # colicTest()

    multiTest()


