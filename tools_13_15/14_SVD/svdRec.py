#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from numpy import *
from numpy import linalg as la

# U, Sigma, VT = linalg.svd([[1, 1], [7, 7]])
# print('U:', U)
# print('Sigma:',Sigma)  # 除了对角元素，其他均为0，Numpy的内部机制产生的仅返回对角元素的方式能节省空间，但是要知道它是一个矩阵
# print('VT:', VT)

def loadExData():
    return [[1, 1, 0, 2, 2],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]
            ]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def euclidSim(inA, inB):
    '''
    欧式距离
    :param inA:列向量
    :param inB: 列向量
    :return:
    '''
    return 1.0/(1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    # 检查是否存在3个或者更多的点，如果不存在，该函数返回1.0，因为此时两个向量完全相关
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    '''
    余弦相似度
    :param inA:
    :param inB:
    :return:
    '''
    num = sum(multiply(inA, inB))
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


def standEst(dataMat, user, simMeas, item):
    '''
    在给定相似度算法的情况下，用户对给定物品的估计评分值
    :param dataMat:矩阵
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 物品编号
    :return:
    '''
    n =shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # 寻找用户都评级的两个物品
        # print(item, j)
        # print('1', logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))
        # print(nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0)))
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]

        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    # 构建对角矩阵
    Sig4 = mat(eye(4)*Sigma[:4])
    print(Sig4)
    # 构建转换后的物品
    xformedItems = dataMat.T*U[:, :4]*Sig4.I
    print(xformedItems)
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating ==0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal



def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # print(dataMat[user, :].A == 0)    # [[False  True  True False False]]
    # print(nonzero(dataMat[user, :].A == 0))  # (array([0, 0], dtype=int64), array([1, 2], dtype=int64))
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # find unrated items
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1, end='')
            else:
                print(0, end='')
        print('')


def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print(shape(myMat))
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print(shape(reconMat))
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)


if __name__ == '__main__':
    # Data = loadExData()
    # myMat = mat(Data)
    # U, Sigma, VT = la.svd(Data)
    # print('Sigma:', Sigma)
    # Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # print(U[:, :3]*Sig3*VT[:3, :])

    # print('euclidSim:', euclidSim(myMat[:, 0], myMat[:, 4]))
    # print('pearsSim:', pearsSim(myMat[:, 0], myMat[:, 4]))
    # print('cosSim:', cosSim(myMat[:, 0], myMat[:, 4]))

    # myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    # myMat[3, 3] = 2
    # print('myMat: ', myMat)
    # print(recommend(myMat, 2))
    # print(recommend(myMat, 2, simMeas=euclidSim))
    # print(recommend(myMat, 2, simMeas=pearsSim))

    # a = [[False, True, True, False, False]]
    # print(nonzero(a))

    # U, Sigma, VT = la.svd(mat(loadExData2()))
    # print('Sigma:', Sigma)
    # Sig2 = Sigma ** 2
    # energy = sum(Sig2)
    # print(energy*0.9)
    # print(sum(Sig2[:2]))
    # print(sum(Sig2[:3]))

    # myMat = mat(loadExData2())
    # print(recommend(myMat, 1, estMethod=svdEst))

    imgCompress(2)

