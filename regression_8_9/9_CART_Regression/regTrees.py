# -*- coding: utf-8 -*-

from numpy import *
import numpy as np

def loadDataSet(filename):
    # 加载数据集
    '''
    :param filename:
    :return: 数据+标签列表
    '''
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def plot_db(dataSet):
    db = array(dataSet)
    x = db[:, -2]
    y = db[:, -1]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c='r')
    plt.xlim([-0.2, 1.2])
    plt.ylim([-1.0, 2.0])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def binSplitDataSet(dataSet, feature, value):
    '''

    :param dataSet:数据集合
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return: 通过数组过滤的方式将上述数据集合切分得到两个子集并返回
    '''
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    '''
    生成叶节点
    :param dataSet: 数据集
    :return:
    '''
    return mean(dataSet[:, -1])


def regErr(dataSet):
    '''
    误差估计函数，在给定数据上计算目标变量的平方误差，总方差
    :param dataSet:
    :return:
    '''
    return var(dataSet[:, -1])*shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''
    找到数据的最佳二元切分方式
    :param dataSet: 数据集
    :param leafType:建立叶节点的函数
    :param errType: 误差计算函数
    :param ops: 包含树构建所需其他参数的元组
    :return: 特征编号和切分特征值
    '''
    tolS = ops[0]  # 容许的误差下降值
    tolN = ops[1]  # 切分的最少样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)  # 当前数据集的大小
    S = errType(dataSet)   # 当前数据集的误差
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0])<tolN or shape(mat1)[0]<tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestS = newS
                bestValue = splitVal
    # 切分后如果误差减少不大，则不应该进行切分操作，直接创建叶节点
    if S-bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 切分后的两个子集大小是否小于用户自定义的大小，则不应切分
    if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    递归函数：树构建
    :param dataSet:数据集
    :param leafType:建立叶节点函数
    :param errType: 误差计算函数
    :param ops: 包含树构建所需的其他函数的元组（tolS，tolN）， tolS：容许的误差下降值，tolN：切分的最少样本数
    :return:
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 满足停止条件时返回叶节点值
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    '''
    测试输入变量是否是一棵树，判断当前处理的节点是否是叶节点
    :param obj:
    :return: 布尔类型
    '''
    return type(obj).__name__ == 'dict'


def getMean(tree):
    '''
    从上往下遍历树直至找到叶节点为止，递归
    :param tree:
    :return: 找到两个叶节点则计算他们的平均值（塌陷处理）
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    '''

    :param tree:待剪枝的树
    :param testData: 剪枝所需的测试数据
    :return:
    '''
    # 没有测试数据则对树进行塌陷处理
    if shape(testData)[0] == 0:
        return getMean(tree)

    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1]-tree['left'], 2)) + sum(power(rSet[:, -1]-tree['right'], 2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1]-treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging!')
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet):
    '''
    将数据集格式化成目标变量Y和自变量X，X和Y用于执行简单的线性回归
    :param dataSet:
    :return: ws,   X, Y
    '''
    m, n = shape(dataSet)
    x = mat(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1: n] = x[:, 0: n-1]
    Y = x[:, -1]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0:
        print('no inverse!')
        raise NameError('This matrix is singular, cannot do inverse,\n\
                try increasing the second value of ops')
    ws = xTx.I * (X.T*Y)
    return ws, X, Y


def modelLeaf(dataset):
    '''
    当数据不再需要切分的时候，负责生成叶节点，返回回归系数ws
    :param dataset:
    :return: 回归系数ws
    '''
    ws, X, Y = linearSolve(dataset)
    # print(ws)
    return ws


def modelErr(dataset):
    '''
    计算给定数据集上的误差
    :param dataset:
    :return:
    '''
    ws, X, Y = linearSolve(dataset)
    yHat = X * ws
    # print(yHat)
    return sum(power(Y-yHat, 2))


def regTreeEval(model, inDat):
    '''
    要对回归树叶节点进行预测
    :param model: ws
    :param inDat:
    :return:
    '''
    return float(model)


def modelTreeEval(model, inDat):
    '''
    对模型树节点进行预测，对数据进行格式化处理，增加第0列，然后计算返回预测值
    :param model: ws
    :param inDat: 输入X矩阵
    :return:
    '''
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1: n+1] = inDat
    return float(X * model)


def treeForeCaast(tree, inDat, modelEval = regTreeEval):
    '''
    对于输入的单个数据点或者行向量，返回一个浮点值，在给定树结构的情况下，对单个数据点，给出一个预测值。
    :param tree: 树
    :param inDat: 要预测的数据点
    :param modelEval: 树结构
    :return:
    '''
    if not isTree(tree):
        return modelEval(tree, inDat)
    if float(inDat[:, tree['spInd']]) > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCaast(tree['left'], inDat, modelEval)
        else:
            # print(modelEval)
            return modelEval(tree['left'], inDat)
    else:
        if isTree(tree['right']):
            return treeForeCaast(tree['right'], inDat, modelEval)
        else:
            # print(modelEval)
            return modelEval(tree['right'], inDat)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = testData.shape[0]
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, :] = treeForeCaast(tree, mat(testData[i]), modelEval)
    return yHat

if __name__ == '__main__':
    # testMat = mat(eye(4))
    # print(testMat)
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print('mat0:', mat0)
    # print('mat1:', mat1)

    # myDat = loadDataSet('ex00.txt')
    # # plot_db(myDat)
    # myMat = mat(myDat)
    # retTree = createTree(myMat)
    # print('retTree  ', retTree)

    # myDat1 = loadDataSet('ex0.txt')
    # plot_db(myDat1)
    # myMat1 = mat(myDat1)
    # retTree1 = createTree(myMat1)
    # print('retTree1  ', retTree1)

    # retTree_1 = createTree(myMat, ops=(0, 1))
    # print('retTree_1  ', retTree_1)

    # # 停止条件tols对误差的数量级非常敏感
    myDat2 = loadDataSet('ex2.txt')
    myMat2 = mat(myDat2)
    retTree2 = createTree(myMat2, ops=(0, 1))

    # print('retTree2  ', retTree2)
    # print('retTree2_1  ', createTree(myMat2, ops=(10000, 4)))

    # 后剪枝：利用测试集来对树进行剪枝，由于不需要用户指定参数，是一种更理想化的剪枝方法。
    myDatTest = loadDataSet('ex2test.txt')
    myMat2Test = mat(myDatTest)
    print('后剪枝：', prune(retTree2, myMat2Test))

    # # 模型树测试
    # myDat = loadDataSet('exp2.txt')
    # # # plot_db(myDat)
    # myMat = mat(myDat)
    # retTree = createTree(myMat, modelLeaf, modelErr, (1, 10))
    # print('retTree  ', retTree)

    # # 模型树与回归树预测比较
    # trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    # testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    # # # 回归树
    # # myTree = createTree(trainMat, ops=(1, 20))
    # # yHat = createForeCast(myTree, testMat[:, 0])
    # # corr = corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    # # # R^2是指相关系数， 越接近1.0越好
    # # print('R^2:', corr)
    # # 模型树
    # myTreeM = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    # yHat = createForeCast(myTreeM, testMat[:, 0], modelTreeEval)
    # corrM = corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    # print('R^2_M:', corrM)
    # # # 标准的线性回归
    # # ws, X, Y = linearSolve(trainMat)
    # # print('ws:', ws)
    # # yHat = mat(zeros((shape(testMat)[0], 1)))
    # # for i in range(shape(testMat)[0]):
    # #     yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    # # corrL = corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    # # print('R^2_L:', corrL)