#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from numpy import *
import urllib
import json
from math import *

def loadDataSet(filename):
    '''
    加载数据
    :param filename:
    :return:一个包含许多列表的列表
    '''
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataMat.append(list(map(float, curLine)))
    return dataMat


def distEclud(vecA, vecB):
    '''
    计算两个向量的欧式距离
    :param vecA:
    :param vecB:
    :return:
    '''
    return sqrt(sum(power(vecA-vecB, 2)))


def randCent(dataSet, k):
    '''
    构建一个包含k个随机质心的集合
    :param dataSet:
    :param k:
    :return:存储k个质心的矩阵
    '''
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ*random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    '''
    二分均值聚类
    :param dataSet:
    :param k:
    :param distMeas: 计算两个向量的欧式距离
    :param createCent:  构建一个包含k个随机质心的集合
    :return:
    '''
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI< minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        # print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def bikMeans(dataSet, k, disMeas=distEclud):
    m = shape(dataSet)[0]  # shape函数此时返回的是dataSet元祖的行数
    clusterAssment = mat(zeros((m, 2)))  # 创建一个m行2列的矩阵，第一列存放索引值，第二列存放误差，误差用来评价聚类效果
    # 创建一个初始簇
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    # print(centList)
    # print(len(centList))
    for j in range(m):
        clusterAssment[j, 1] = disMeas(mat(centroid0), dataSet[j, :]) ** 2  # 计算所有点的均值，选项axis=0表示沿矩阵的列方向进行均值计算
    while len(centList) < k:
        lowestSSE = inf  # inf正无穷大
        for i in range(len(centList)):
            # 尝试划分每一簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, disMeas)

            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit and notSplit:", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

        print("the bestCentToSplit is :", bestCentToSplit)
        print("the len of bestClustAss is:", len(bestClustAss))
        # 更改i的簇心
        centList[bestCentToSplit] = bestNewCents[0, :]
        # 增加新的质心
        centList.append(bestNewCents[1, :])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment
    # return mat(centList),clusterAssment

#
# def geoGrab(stAddress, city):
#     apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
#     params = {}
#     params['flags'] = 'J'  # JSON return type
#     params['appid'] = 'aaa0VN6k'
#     params['location'] = '%s %s' % (stAddress, city)
#     url_params = urllib.urlencode(params)
#     yahooApi = apiStem + url_params  # print url_params
#     print(yahooApi)
#     c = urllib.urlopen(yahooApi)
#     return json.loads(c.read()
#
# from time import sleep
# def massPlaceFind(fileName):
#     fw = open('places.txt', 'w')
#     for line in open(fileName).readlines():
#         line = line.strip()
#         lineArr = line.split('\t')
#         retDict = geoGrab(lineArr[1], lineArr[2])
#         if retDict['ResultSet']['Error'] == 0:
#             lat = float(retDict['ResultSet']['Results'][0]['latitude'])
#             lng = float(retDict['ResultSet']['Results'][0]['longitude'])
#             print
#             "%s\t%f\t%f" % (lineArr[0], lat, lng)
#             fw.write('%s\t%f\t%f\n' % (line, lat, lng))
#         else:
#             print
#         "error fetching"
#         sleep(1)
#     fw.close()


def distSLC(vecA, vecB):
    '''
    返回地球表面两点间的距离（单位：英里）
    :param vecA:
    :param vecB:
    :return:
    '''
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return math.arccos(a + b) * 6371.0  # pi is imported with numpy


import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []  # 样本
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])  # 保存经纬度
    datMat = mat(datList)   # 数据集 numpy的mat类型
    # 进行二分K均值算法聚类
    myCentroids, clustAssing = bikMeans(datMat, numClust, disMeas=distSLC)
    fig = plt.figure()   # 窗口
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)  # 轴
    imgP = plt.imread('Portland.png')  # 标注在实际的图片上
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):  # 每一个中心
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A == i)[0], :]  # 属于每个中心的样本点
        markerStyle = scatterMarkers[i % len(scatterMarkers)]  # 点的类型 画图
        # 散点图 每个中心的样本点
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    # print(array(myCentroids)[:, :, 0].flatten().tolist())
    # 散 点图 每个中心
    ax1.scatter(array(myCentroids)[:, :, 0].flatten().tolist(), array(myCentroids)[:, :, 1].flatten().tolist(), marker='+', s=300)
    plt.show()


if __name__ == '__main__':
    # print(3*random.rand(3, 1))
    # data = loadDataSet('testSet.txt')
    # dataMat = mat(data)
    # print(min(dataMat[:, 0]))
    # print(max(dataMat[:, 0]))
    # print(min(dataMat[:, 1]))
    # print(max(dataMat[:, 1]))
    # centroids = randCent(dataMat, 3)
    # print(centroids)
    # print(distEclud(dataMat[0], dataMat[1]))

    # myCentroids, clusteAssing = kMeans(dataMat, 4)
    # print('center:', myCentroids)
    # print('clusterAssing', clusteAssing)

    # data1 = loadDataSet('testSet2.txt')
    # dataMat1 = mat(data1)
    # cenList, myNewAssment = bikMeans(dataMat1, 3)
    # print('cenList:', cenList)

    clusterClubs(5)

