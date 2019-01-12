# -*- coding: utf-8 -*-
"""
@author: WF
"""
import numpy as np
import operator

# 创建数据集
def createDataSet():
    group = np.array([[1.0, 1.0], [1.1, 1.0], [0, 0.1], [0, 0]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#分类
'''
1 计算已知类别数据集中的每个样本点到当前点的距离
2 按照距离递增次序排序
3 选取与当前点距离最近的k个点
4 确定前k个点所在类别的出现频率
5 返回前k个点出现频率最高的类别作为当前点的预测分类
'''
def classify0(inX, dataset, labels, k):
    '''
    :param inX: 用于分类的输入向量
    :param dataset: 训练数据集向量
    :param labels: 标签向量
    :param k: 选择最近邻的数目
    :return: 类别
    '''
    dataset_size = dataset.shape[0]
    diffMat = np.tile(inX, (dataset_size, 1))-dataset
    sqDiffMat = diffMat**2
    sqdistances = sqDiffMat.sum(axis=1)
    distances = sqdistances**0.5
    k_index = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIndex = labels[k_index[i]]
        classCount[voteIndex] = classCount.get(voteIndex, 0)+1
    sortClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]

def file2matrix(filename):
    '''
    将文件解析成矩阵的形式
    :param filename: 要解析的文件名
    :return: 数据矩阵和标签向量
    '''
    f = open(filename, 'r')
    f_readlines = f.readlines()
    data_size = len(f_readlines)
    dataSet = np.zeros((data_size, 3))
    labels = []
    num = 0
    for line in f_readlines:
        line = line.strip().split('\t')
        dataSet[num, :] = line[0:3]
        labels.append(int(line[-1]))
        num += 1
    return dataSet, labels


def autoNorm(dataSet):
    '''
    归一化特征值
    nexValue = (oldValue-min)/(max-min)
    :param dataSet: 数据集
    :return: 将数字特征值转化为0到1区间,归一化的矩阵，max-min, minvals
    '''
    minvals = dataSet.min(0) # 从列中选取最小值
    maxvals = dataSet.max(0)
    ranges = maxvals - minvals
    norm_dataset = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    norm_dataset = dataSet - np.tile(minvals, (m, 1))
    norm_dataset = norm_dataset/np.tile(ranges, (m, 1))
    return norm_dataset, ranges, minvals


def datingClassTest():
    '''
    前百分之十用于测试，其余数据用于训练分类器
    :return: None
    '''
    horatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(horatio*m)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real label is :%d' %(classifierResult, datingLabels[i]))
        if(classifierResult!= datingLabels[i]):
            errorCount += 1.0
    print('the total error rate is %f' %(errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    icecream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minvals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,icecream])
    classifierResult = classify0((inArr-minvals)/ranges, normMat, datingLabels, 3)
    print('you will probably like this person:', resultList[classifierResult-1])

import matplotlib.pyplot as plt


def draw_scatter(datingLabels):
    from matplotlib.font_manager import FontProperties
    zhfont = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=20)  # 指定本机的汉字字体位置
    fig = plt.figure(figsize=(16, 10), dpi=80) # figsize为画布尺寸，dpi是像素点
    ax1 = fig.add_subplot(221) # 子图总行数、总列数、位置
    ax1.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    plt.xlabel(u'玩视频游戏所耗费的时间百分比', fontproperties=zhfont)
    plt.ylabel(u'每周消费的冰淇淋公斤数', fontproperties=zhfont)
    ax2 = fig.add_subplot(222)
    ax2.scatter(datingDataMat[:, 1], datingDataMat[:, 2], s=10.0 * np.array(datingLabels), c=10.0 * np.array(datingLabels))
    plt.xlabel(u'玩视频游戏所耗费的时间百分比', fontproperties=zhfont)
    plt.ylabel(u'每周消费的冰淇淋公斤数', fontproperties=zhfont)
    ax3 = fig.add_subplot(223)
    ax3.scatter(datingDataMat[:, 0], datingDataMat[:, 1], s=15.0 * np.array(datingLabels), c=15.0 * np.array(datingLabels))
    plt.xlabel(u'每年获得的飞行常客里程数', fontproperties=zhfont)
    plt.ylabel(u'玩视频游戏所耗费的时间百分比', fontproperties=zhfont)
    ax4 = fig.add_subplot(224)
    datingLabels = np.array(datingLabels)
    idx_1 = np.where(datingLabels == 1)
    p1 = ax4.scatter(datingDataMat[idx_1, 0], datingDataMat[idx_1, 1], marker='*', color='r', label='1',s=10)
    idx_2 = np.where(datingLabels == 2)
    p2 = ax4.scatter(datingDataMat[idx_2, 0], datingDataMat[idx_2, 1], marker='o', color='g', label='2',s=20)
    idx_3 = np.where(datingLabels == 3)
    p3 = ax4.scatter(datingDataMat[idx_3, 0], datingDataMat[idx_3, 1], marker='+', color='b', label='3',s=30)
    plt.xlabel(u'每年获得的飞行常客里程数', fontproperties=zhfont)
    plt.ylabel(u'玩视频游戏所耗费的时间百分比', fontproperties=zhfont)
    ax4.legend((p1, p2, p3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2, prop=zhfont)
    # plt.legend('upper left')
    plt.show()

    # # 将三类数据分别取出来
    # # x轴代表飞行的里程数
    # # y轴代表玩视频游戏的百分比
    # type1_x = []
    # type1_y = []
    # type2_x = []
    # type2_y = []
    # type3_x = []
    # type3_y = []
    #
    # for i in range(len(labels)):
    #     if labels[i] == 1:  # 不喜欢
    #         type1_x.append(matrix[i][0])
    #         type1_y.append(matrix[i][1])
    #
    #     if labels[i] == 2:  # 魅力一般
    #         type2_x.append(matrix[i][0])
    #         type2_y.append(matrix[i][1])
    #
    #     if labels[i] == 3:  # 极具魅力
    #         # print (i, '：', labels[i], ':', type(labels[i]))
    #         type3_x.append(matrix[i][0])
    #         type3_y.append(matrix[i][1])
    #
    # type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    # type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
    # type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')
    #
    # plt.xlabel(u'每年获取的飞行里程数', fontproperties=zhfont)
    # plt.ylabel(u'玩视频游戏所消耗的事件百分比', fontproperties=zhfont)
    # axes.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2, prop=zhfont)
    # plt.show()


if __name__ == '__main__':
    data, labels = createDataSet()
    print(classify0([0.2, 0], data, labels, 3))

    # from imp import reload
    # reload(kNN)
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # print(np.array(datingLabels[:5]))
    # print(datingDataMat[:10])
    # print(datingLabels[:10])

    draw_scatter(datingLabels)

    # data = np.array([[  4.09200000e+04,8.32697600e+00,9.53952000e-01],
    #                  [  1.44880000e+04,7.15346900e+00,1.67390400e+00],
    #                  [  2.60520000e+04,1.44187100e+00,8.05124000e-01],
    #                  [  7.51360000e+04,1.31473940e+01,4.28964000e-01],
    #                  [  3.83440000e+04,1.66978800e+00,1.34296000e-01]])
    # labels = np.array([3,2,1,1,1])
    # print(data,labels)
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # ax.scatter(data[:,0],data[:,1],s=15*np.array([3,2,1,1,1]), c=10*np.array([3,2,1,1,1]))
    # plt.show()

    normMat, ranges, minvals = autoNorm(datingDataMat)
    print(normMat)

    # datingClassTest()

    classifyPerson()