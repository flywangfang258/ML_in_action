#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import random
import numpy as np


def loadDataSet(filename):
    '''
    加载数据集
    :param filename:
    :return: 数据矩阵，标签列表
    '''
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


def selectJrand(i, m):
    '''
    # 该辅助函数用于在某个区间范围内随机选择一个整数
    :param i:第一个alpha的下标
    :param m: 所有alpha的数目
    :return:
    '''
    j = i
    while(j==i):
        j = int(random.uniform(0, m))  # random.uniform(0,m)用于生成指定范围内的随机浮点数
    return j


def clipAlpha(aj, H, L):
    '''
    用于调整大于H或者小于L的alpha值, 该辅助函数用于在数值太大时对其进行调整
    :param aj:
    :param H:
    :param L:
    :return:
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    # 简化版SMO算法
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前最大的循环次数
    :return:
    :fXi: 预测的类别
    :Ei: 误差
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()  # 转置之前是列表，转置后是一个列向量
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0  # 该变量存储的是在没有任何alpha改变时遍历数据集的次数
    while(iter < maxIter):  # 限制循环迭代次数，也就是在数据集上遍历maxIter次，且不再发生任何alpha修改，则循环停止
        alphaPairsChanged = 0  # 每次循环时先设为0，然后再对整个集合顺序遍历，该变量用于记录alpha是否已经进行优化
        for i in range(m):  # 遍历每行数据向量，m行
            # 该公式是分离超平面，我们预测值
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)+b)
            print('fxi:', fXi)
            # 预测值和真实输出之差
            Ei = fXi - float(labelMat[i])
            # 如果误差很大就对该数据对应的alpha进行优化，正负间隔都会被测试，同时检查alpha值
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                j = selectJrand(i, m)  # 随机选择不等于i的0-m的第二个alpha值
                fXj = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                #保证alpha在0和C之间
                if labelMat[i] != labelMat[j]:  # 这里是对SMO最优化问题的子问题的约束条件的分析
                    L = max(0, alphas[j] - alphas[i])  # L和H分别是alpha所在的对角线端点的界
                    H = min(C, C+alphas[j]-alphas[i])  # 调整alphas[j]位于0到c之间
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L == H:
                    print('L==H')
                    continue   # L=H停止本次循环
                # eta是一个中间变量：eta=2xi*xj-xixi-xjxj，是alphas[j]的最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * \
                      dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:  # eta>=0停止本次循环，这里是简化计算
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta  # 沿着约束方向未考虑不等式约束时的alpha[j]的解
                alphas[j] = clipAlpha(alphas[j], H, L)   # 此处是考虑不等式约束的alpha[j]解
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")  # 如果该alpha值不再变化，就停止该alpha的优化
                    continue
                # 更新alpha[i]
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # update i by the same amount as j
                # the update is in the oppostie direction
                # 完成两个alpha变量的更新后，都要重新计算阈值b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0   # alpha[i]和alpha[j]是0或者c,就取中点作为b
                alphaPairsChanged += 1   # 到此的话说明已经成功改变了一对alpha
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            if (alphaPairsChanged == 0):
                iter += 1   # 如果alpha不再改变迭代次数就加1
            else:
                iter = 0
            print("iteration number: %d" % iter)
        return b, alphas


# 完整版platt SMO 算法的支持函数
# 建立一个数据结构来保存所有的重要值
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]  # 有多少行数据
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 误差缓存，第一列是ecache是否有效的标志位，第二列是实际的E值

# 计算E值并返回,E值是函数对输入xi的预测值与真实输出的差
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+ oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):     # 该函数的误差值与第一个alpha值Ei和下标i有关
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  # 设置有效，有效意味着它已经计算好了
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0] # 构建出一个非零表，返回的列表中包含以输入列表为目录的列表值
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue # 跳出本次循环
            Ek = calcEk(oS, k) # 传递对象和k，计算误差值
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):  # 选择具有最大步长的j
                maxK = k; maxDeltaE = deltaE; Ej = Ek # 会在所有的值上循环，并选择其中使得改变最大的那个值
        return maxK, Ej
    else:   # 在这种情况下（第一次，我们没有任何有效的eCache值 ），随机选择一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k): # alpha改变时更新缓存中的值
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]


# 完整platt SMO算法中的优化例程
def innerL(i, oS):
    Ei = calcEk(oS, i)  # 计算误差值
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
           ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)  # 第二个alpha选择中的启发式方法
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j) # 更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i, :]*oS.X[i, :].T \
             - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i, :]*oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i, :]*oS.X[j, :].T \
             - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j, :]*oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1  # 如果有任意一对alpha发生改变，那么就会返回1，其他返回0

    else:
        return 0



# 完整版platt SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter):
    # 建立一个数据结构来容纳所有的数据
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0  # 退出循环的变量的一些初始化
    # 迭代次数超过指定的最大值或者遍历整个集合都未对任意的alpha对进行修改时就退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):  # 一开始在数据集上遍历任意可能的alpha
                # 选择第二个alpha，并在可能时对其进行优化处理,有任一一对alpha发生变化化了alphaPairsChanged+1
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:   # 遍历所有的非边界alpha值，也就是不在边界0或c上的值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False  # 在非边界循环和完整遍历之间进行切换
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


# 分类超平面的w计算
def calcWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# 核函数的smop
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def kernelTrans(X, A, kTup):
    '''
     #calc the kernel or transform data to a higher dimensional space
    :param X:
    :param A:
    :param kTup: kernel information (,,)
    :return:
    '''
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We have a problem -- That kernel is not recognized!')
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2))) #first column is valid flag
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!= np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))


def img2vector(filename):
    '''
    把32*32的二进制图像矩阵转换为1*1024的向量
    :param filename:
    :return:
    '''
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    # print(labelMat)

    # b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    # print('b:', b)
    # print('alphas[alphas>0]:', alphas[alphas>0])  # 数组过滤
    # print(np.shape(alphas[alphas>0]))  # 得到支持向量的个数
    # for i in range(100):  # 得到是支持向量的数据点
    #     if alphas[i]>0.0:
    #         print(dataMat[i], labelMat[i])



    # dataArr, labelArr = loadDataSet('testSet.txt')
    # b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # print('b:', b)
    #
    # print('alphas:', alphas)   # 输出w和b
    #
    # ws = calcWs(alphas, dataArr, labelArr)
    # print('ws:', ws)
    #
    # datmat = np.mat(dataArr)
    # result = datmat[0] * np.mat(ws) + b  # 进行分类
    # print('result', result)

    # testRbf()

    testDigits(kTup=('rbf', 20))

    # x = np.mat([[1, 2], [3, 4], [5, 6]])
    # a = np.mat([[1], [2], [3]])
    # print(a)
    # y = np.mat([1, 0, 1]).transpose()
    # print('y:', y)
    # z = np.multiply(a, y)
    # b = x * x[0, :].T
    # print('z: ', z)
    # print('b: ', b)

    # x = np.mat(np.zeros((3, 2)))
    # print('x:', x)
    # print(np.nonzero(x[:, 0].A))
    # # 转成array
    # print(np.nonzero(x[:, 0].A)[0])
    # print(x[:, 0].A)


