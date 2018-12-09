#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

# 数组
from numpy import array
mm = array([1, 1, 1])
print(mm)  # [1 1 1]
pp = array((1, 2, 3))
print(mm + pp)  # [2 3 4]
print(mm*2)   # [2 2 2]
# 多维数组中的元素可以像列表中一样访问，也可以用矩阵方式访问
jj = array([[1, 2, 3], [1, 1, 1]])
print(jj[0][2], jj[0, 2])  # 3 3

# 矩阵
from numpy import mat, matrix
ss = mat([1, 5, 3])
print('ss:', ss)    # ss: [[1 5 3]]
mm = matrix([7, 2, 3])
print('mm:', mm)   # mm: [[7 2 3]]
print(mm[0, 1])  # 2

# 把python列表转换成Numpy矩阵
pyList = [4, 5, 7]
print(mat(pyList))  # [[4 5 7]]
# print(mm*ss)  # ValueError: shapes (1,3) and (1,3) not aligned: 3 (dim 1) != 1 (dim 0) 矩阵乘法，注意维度
print(mm*ss.T)  # [[26]]

# 查看矩阵或者数组的维度
from numpy import shape
print(shape(mm))   # (1, 3)

# 把矩阵mm的每个元素和矩阵ss的每个元素对应相乘，使用multiply
from numpy import multiply
print(multiply(mm, ss))  # [[ 7 10  9]]

# 矩阵和数组还有很多方法，如排序
mm.sort()  # 原地排序，排序后结果占用原始空间，如果需要保留数据的原序，必须先做一份拷贝，或者使用argsort
print(mm)  # [[2 3 7]]
print(ss.argsort())   # [[0 2 1]]
for i in ss.argsort().A[0].tolist():
    print(ss[0, i])

print(mm.mean())   # 4.0

print(jj[1, :])  # [1 1 1]
print(jj[0, 0:2])
