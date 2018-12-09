#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'
# import warnings
import numpy as np
# warnings.simplefilter(action='ignore', category=FutureWarning)

def loadSimpleData():
    dataMat = np.matrix([[1.,  2.1],
                         [2.,  1.1],
                         [1.3,  1.],
                         [1.,  1.],
                         [2.,  1.]])
    classLabel = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabel


def adaptive_load_data(filename):
    numFeat = len(open(filename).readline().strip().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plot_simpledata():
    import matplotlib.pyplot as plt
    x, y = loadSimpleData()
    y = np.array(y)
    x = np.array(x)
    # print(y)
    # x1 = x[np.nonzero(y)[0]]
    x1 = x[np.where(y == 1.0)[0]]
    # print(np.nonzero(y)[0])
    x0 = x[np.where(y == -1.0)[0]]
    # print(x0[:, 0], x0[:, 1])
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(x0[:, 0], x0[:, 1], marker='<', c='r', s=50)
    ax.scatter(x1[:, 0], x1[:, 1], marker='s', c='g', s=20)
    plt.show()


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对数据进行分类，所有在阈值一边的数据会被分到-1，另一边的数据被分到1
    :param dataMatrix:数据矩阵
    :param dimen: 维度属性
    :param threshVal: 阈值
    :param threshIneq:阈值比较符号
    :return:单层决策树字典，错误率，类别估计
    '''
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    # classLabels = np.reshape(classLabels, (len(classLabels), 1))

    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    # print(np.shape(labelMat))   # (5,1)
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # print('p:', predictedVals)
                # print(np.shape(predictedVals))  #(5, 1)
                errArr = np.mat(np.ones((m, 1)))
                # print(np.where(predictedVals == labelMat)[0])
                # for num in range(len(predictedVals)):
                #     print(predictedVals[num] == classLabels[num])
                #     if float(predictedVals[num]) == float(classLabels[num]):
                #         errArr[num][0] = 0
                # print(errArr)
                errArr[predictedVals == labelMat] =0
                # print(errArr)
                weightedError = D.T * errArr   # 计算加权错误率
                print("split:dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" \
                      %(i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D:', D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:', classEst.T)
        # 为下一次迭代计算D
        expon = np.multiply(-1*alpha*np.mat(list(map(float, classLabels))).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print('aggClassEst:', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print('errorRate:', errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print('aggClassEst:', aggClassEst)
    return(np.sign(aggClassEst))

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0   # 用于计算AUC的值
    numPosClas = sum(np.array(classLabels) == 1.0)   # 计算正例的数目
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()   # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area Under the Curve is:', ySum*xStep)


if __name__ == "__main__":
    dataMat, classLabels = loadSimpleData()
    # plot_simpledata()

    # D = np.mat(np.ones((5, 1))/5)
    # bestStump, minError, bestClasEst = buildStump(dataMat, classLabels, D)
    # print(bestStump, minError, bestClasEst)

    # weakClassArr = adaBoostTrainDS(dataMat, classLabels, 9)
    # print(weakClassArr)
    #
    # print(adaClassify([[0, 0], [5, 5]], weakClassArr))

    datArr, labelArr = adaptive_load_data('horseColicTraining2.txt')
    # print(datArr)
    print(labelArr)
    classifierArr, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
    # testArr, testLabelArr = adaptive_load_data('horseColicTest2.txt')
    # prediction10 = adaClassify(testArr, classifierArr)
    # print(prediction10.T)
    # errArr = np.mat(np.ones((67, 1)))
    # errnum = errArr[prediction10 != np.mat(testLabelArr).T].sum()
    # print('errornum:', errnum, 'errorRate:', errnum/67)

    plotROC(aggClassEst.T, labelArr)