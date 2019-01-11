#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from math import log
import numpy as np
import operator

# step1 计算给定数据的熵
def calShannonEnt(dataset):
    # dataset 为list  并且里面每一个list的最后一个元素为label
    # 如[[1,1,'yes'],
    #    [1,1,'yes'],
    #    [1,0,'no'],
    #    [0,0,'no'],
    #    [0,1,'no']]

    numClass = len(dataset)  # 获得list的长度 即实例总数 注（a）若为矩阵，则 len(dataset.shape[0])
    e_class = {}  # 创建一个字典，来存储数据集合中不同label的数量 如 dataset包含3 个‘yes’  2个‘no’ （用键-值对来存储）
    #遍历样本统计每个类中样本数量
    for example in dataset:
        if example[-1] not in e_class.keys(): # 如果当前标签在字典键值中不存在
            e_class[example[-1]] = 0 # 初值
        e_class[example[-1]] += 1 # 若已经存在 该键所对应的值加1
    shannonEnt = 0.0
    for k in e_class.keys():
        prob = e_class[k]/numClass   # 计算单个类的熵值
        shannonEnt -= prob * log(prob, 2)  # 累加每个类的熵值
    return shannonEnt


# step2 计算信息增益 （判断哪个属性的分类效果好）

# # 以属性i，value划分数据集
def split_dataset(dataset, i, value):
    ret_dataset = []
    for example in dataset:
        if example[i] == value:  # 将符合特征的数据抽取出来 比如 属性wind=｛weak，strong｝ 分别去抽取: weak多少样本，strong多少样本
            ret_feature = example[:i]  # 0-(attribute-1)位置的元素
            ret_feature.extend(example[i+1:])  # 去除了 attribute属性
            ret_dataset.append(ret_feature)
    return ret_dataset   # 返回 attribbute-{A}

def choseBestFeature(dataset):    # 选择最优的分类特征
    feature_count = len(dataset[0]) - 1
    baseEnt = calShannonEnt(dataset)  # 原始的熵
    best_gain = 0.0
    best_feature = -1
    for i in range(feature_count):
        #  python中的集合(set)数据类型，与列表类型相似，唯一不同的是set类型中元素不可重复
        unique_feature = set([example[i] for example in dataset])
        new_entropy = 0.0
        for value in unique_feature:
            sub_dataset = split_dataset(dataset, i, value)  # 调用函数返回属性i下值为value的子集
            prob = len(sub_dataset)/len(dataset)
            new_entropy += prob * calShannonEnt(sub_dataset)  # 计算每个类别的熵
        info_gain = baseEnt - new_entropy  # 求信息增益
        if best_gain < info_gain:
            best_gain = info_gain
            best_feature = i
    return best_feature  # 返回分类能力最好的属性索引值



def createTree(dataset, attribute):
    class_lable = [example[-1] for example in dataset]  # 类别：男或女
    if class_lable.count(class_lable[0]) == len(class_lable):
        return class_lable[0]
    if len(dataset[0]) == 1:
        return majority_count(class_lable)
    best_feature_index = choseBestFeature(dataset)  # 选择最优特征
    best_feature = attribute[best_feature_index]
    my_tree = {best_feature: {}}  # 分类结果以字典形式保存
    del(attribute[best_feature_index])
    feature_value = [example[best_feature_index] for example in dataset]
    unique_f_value = set(feature_value)
    for value in unique_f_value:
        sublabel = attribute[:]
        my_tree[best_feature][value] = createTree(split_dataset(dataset, best_feature_index, value), sublabel)
    return my_tree



def majority_count(classlist):
    class_count = {}
    for vote in classlist:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_class_count) # [('yes', 3), ('no', 2)]
    return sorted_class_count[0][0]


def createDataSet1():  # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发', '声音']  # 两个特征
    return dataSet, labels


if __name__ == "__main__":
    dataset = [[1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,0,'no'],
        [0,1,'no']]
    print(len(dataset))
    print(calShannonEnt(dataset))

    # dataset = np.array(dataset)
    # print(dataset.shape[0])
    # print(calShannonEnt(dataset))

    # classlist = ['yes', 'yes', 'no', 'no', 'yes']
    # print(majority_count(classlist))
    dataSet, labels = createDataSet1()  # 创造示列数据
    print(createTree(dataSet, labels))  # 输出决策树模型结果

