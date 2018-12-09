#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from numpy import *


def loadDataSet(filename, delim = '\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    # 去平均值
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals   # 每一列都减去均值
    covMat = cov(meanRemoved, rowvar=0)   # 计算新矩阵协方差
    # print(covMat)
    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算新矩阵特征值和特征向量
    # print(eigVals, eigVects)
    eigValInd = argsort(eigVals)   # 将N个特征值从小到大排序  argsort函数返回的是数组值从小到大的索引值
    # print('eigValInd: ', eigValInd)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # print(eigValInd)
    redEigVects = eigVects[:, eigValInd]
    # 将数据转换到新空间
    # print(shape(meanVals), shape(redEigVects))
    lowDDataMat = meanRemoved*redEigVects   # 降到1维特征空间的数据
    reconMat = (lowDDataMat*redEigVects.T)+meanVals  # 重构数据
    # print('将样本点投影到选取的低维特征向量上,实际使用的是这个结果作为新的特征：\n', lowDDataMat)
    # print('重构后的样本:\n', reconMat)
    return lowDDataMat, reconMat


def plot(dataMat, reconMat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=30, c='red')
    plt.show()



def replaceNanWithMean():
    '''
    将NaN替换成平均值
    :return:
    '''
    datMat = loadDataSet('secom.data', ' ')
    # 特征数目
    numFeat = shape(datMat)[1]
    print(numFeat)
    for i in range(numFeat):
        # 计算所有非NaN的平均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # 将所有NaN置为平均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


if __name__ == '__main__':
    # dataMat = loadDataSet('testSet.txt')
    # lowDMat, reconMat = pca(dataMat, 1)
    # print(shape(lowDMat))
    # plot(dataMat, reconMat)

    # a = [1, 2, 3, 4, 5, 6, 7]
    # print(a[:-3:-1])
    # print(a[:-3:-2])


    # from sklearn.decomposition import PCA
    #
    # # 原始数据
    # data = array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7],[2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])
    #
    # pca = PCA(n_components=1)
    # new_feature = pca.fit_transform(data)
    # print(new_feature)

    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(covMat)
    print('eigVals: ', len(eigVals), len(eigVals[nonzero(eigVals)[0]]), (len(eigVals)- len(eigVals[nonzero(eigVals)[0]]))/len(eigVals))

    # a = array([1, 2, 3, 4, 0, 0])
    # print(len(a), nonzero(a))   # 6 (array([0, 1, 2, 3], dtype=int64),)
    # print(nonzero(a)[0])  # [0 1 2 3]
    # print(a[nonzero(a)[0]])  # [1 2 3 4]