#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import numpy as np


def loadDataSet(filename):
    '''
    加载数据集
    :param filename: 文件名
    :return: 数据列表dataMat， 标签列表labelMat
    '''
    dataMat = []
    labelMat = []
    featNum = len(open(filename).readline().strip().split('\t')) - 1
    fr = open(filename)
    for line in fr.readlines():
        dataTemp = []
        curLine = line.strip().split('\t')
        # print(curLine)
        for i in range(featNum):
            dataTemp.append(float(curLine[i]))
        dataMat.append(dataTemp)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def loadDataSet1(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # print(yMat)
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = np.linalg.inv(xTx) * (xMat.T * yMat)
    yHat1 = xMat * ws

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat[:, 0].flatten().A[0])
    # plt.show()

    xCopy = xMat.copy()
    xCopy.sort(0)  # 按列进行排序
    # print(xCopy)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    局部加权线性回归
    :param testPoint: 待测点
    :param xArr:
    :param yArr:
    :param k:
    :return:
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    srtInd = np.array(xArr)[:, 1].argsort(0)
    # print(srtInd)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    xSort = []
    ySort = []
    ySortArr = []
    for i in srtInd:
        xSort.append(np.array(xArr)[i, 1])
        ySort.append(yHat[i])
        ySortArr.append(yArr[i])

    ax = fig.add_subplot(111)
    # print(xSort)
    ax.scatter(np.mat(xArr)[:, 1].flatten().A[0], yArr, s=2, c='red')
    ax.plot(xSort, ySort)
    plt.show()
    return yHat


def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    岭回归
    :param xMat:
    :param yMat:
    :param lam:
    :return:
    '''
    xTx = xMat.T*xMat
    denom = xTx+np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T*yMat)
    return ws


def ridgeTest(xArr, yArr):
    '''
    30个lam进行测试
    :param xArr:
    :param yArr:
    :return:
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 数据标准化
    yMean = np.mean(yMat, 0)
    yMat = yMat-yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat-xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat))[1])
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)   #calc mean then subtract it off
    inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''
    前向逐步回归
    :param xArr:
    :param yArr:
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return:
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


def main():
    xArr,yArr = loadDataSet1('abalone.txt')
    stageWise(xArr, yArr, 0.001, 5000)


from bs4 import BeautifulSoup


def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    函数说明:从页面读取数据，生成retX和retY列表
    Parameters:
        retX - 数据X
        retY - 数据Y
        inFile - HTML文件
        yr - 年份
        numPce - 乐高部件数目
        origPrc - 原价
    Returns:
        无
    """
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)


def setDataCollect(retX, retY):
    """
    函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
    Parameters:
        无
    Returns:
        无
    Website:
        https://www.cuijiahua.com/
    Modify:
        2017-12-03
    """
    scrapePage(retX, retY, './setHtml/lego8288.html', 2006, 800, 49.99)  # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './setHtml/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './setHtml/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './setHtml/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './setHtml/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './setHtml/lego10196.html', 2009, 3263, 249.99)  # 2009年的乐高10196,部件数目3263,原价249.99


def crossValidation(xArr, yArr, numVal=10):
    import random
    m = len(xArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(xArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            # 用训练时的参数将测试数据标准化
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0)
            varTrain = np.var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)  # test ridge results and store
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY).T)
            # print errorMat[i,k]
    meanErrors = np.mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    # print(meanX)
    varX = np.var(xMat, 0)
    # 为了和标准线性回归比较需要对数据进行还原????????????
    unReg = bestWeights / varX
    print(unReg)
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 *(np.multiply(meanX, unReg).sum()) + np.mean(yMat))



if __name__ == "__main__":
    xArr, yArr = loadDataSet("ex0.txt")

    # print(xArr[0:2])
    # print(yArr)
    # ws, yHat = standRegres(xArr, yArr)
    # print('corrcoef:', np.corrcoef(yHat.T, np.mat(yArr)))  # 计算相关系数

    # print('ws: ', ws)
    # print('yHat', yHat)

    # print(np.eye(3))  # 单位矩阵

    # print(lwlr(xArr[0], xArr, yArr, 1.0))
    # print(lwlr(xArr[0], xArr, yArr, 0.001))
    #
    # yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    # print(yHat)

    abX, abY = loadDataSet('abalone.txt')
    # yHat_01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    # print('训练集， k= 0.1:', rssError(abY[0:99], yHat_01.T))
    #
    # yHat_1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    # print('测试集， k= 0.1：', rssError(abY[100:199], yHat_1.T))
    #
    # yHat_1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    # print('训练集， k= 1:', rssError(abY[0:99], yHat_1.T))
    #
    # yHat_1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    # print('测试集， k= 1：', rssError(abY[100:199], yHat_1.T))
    #
    # yHat_10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    # print('训练集， k= 10：', rssError(abY[0:99], yHat_10.T))
    #
    #
    # yHat_10 = lwlrTest(abX[100:199], abX[0:99], abY[100:199], 10)
    # print('测试集， k= 10：', rssError(abY[100:199], yHat_10.T))
    #
    #
    # ws, yHat_line_train = standRegres(abX[0:99], abY[0:99])
    # yHat_line = np.mat(abX[100:199])*ws
    # print('训练集， linear：', rssError(abY[0:99], yHat_line_train.T.A))
    # print('测试集， linear：', rssError(abY[100:199], yHat_line.T.A))


    # ridgeWeights = ridgeTest(abX, abY)
    # # print(abX[4], np.mat(ridgeWeights[0]).T)
    # # print(np.mat(abX[4])*(np.mat(ridgeWeights[0]).T/np.var(abX))+np.mean(abY))
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.xlabel('log(lambda)')
    # # ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties=font)
    # # ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties=font)
    # # ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties=font)
    # # ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    # # plt.setp(ax_title_text, size=20, weight='bold', color='red')
    # # plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    # # plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    # plt.xlim((0, 30))
    # plt.ylabel('ws')
    # plt.show()

    # ws_mat = stageWise(abX, abY, 0.001, 200)
    # print(ws_mat)
    # main()

    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    # print(lgX[0:10])
    print(lgY[0:10])

    lgX1 = np.mat(np.ones((np.shape(lgX)[0], np.shape(lgX)[1]+1)))
    lgX1[:, 1:5] = np.mat(lgX)
    # print(lgX[0], lgX1[0])
    ws = standRegres(lgX1, lgY)
    print(lgX1[0]*ws)
    print(lgX1[-1]*ws)
    print(lgX1[43]*ws)

    crossValidation(lgX, lgY, 10)

    print(ridgeTest(lgX, lgY))