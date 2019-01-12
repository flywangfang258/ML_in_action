# -*- coding: utf-8 -*-
"""
@author: 王方
"""
import numpy as np
import os
import sys
sys.path.append('D:\\pycode\\ML_in_action\\ch01_knn')
from kNN import classify0

# 将图像转为1*1024的数组
def img2vector(filename):
    returnVec = np.zeros((1, 1024))   # 创建1*1024的Numpy数组
    fr = open(filename)     # 打开给定文件
    for i in range(32):  # 循环读出文件的前32行
        line = fr.readline()
        for j in range(32):  # 将每行的前32个字符值存储在Numpy数组中
            returnVec[0,32*i + j] = int(line[j])
    return returnVec


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        classNum = int(trainingFileList[i].split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i,:] = img2vector('trainingDigits/%s' %(trainingFileList[i]))
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        classNumT = int(testFileList[i].split('_')[0])
        vecTest = img2vector('testDigits/%s' %(testFileList[i]))
        classResult = classify0(vecTest, trainingMat, hwLabels, 3)
        print('the classifier came back with %d,the real answer is %d' %(classResult, classNumT))

        if classResult!= classNumT:
            errorCount += 1.0
    print('the total error number is %f' %(errorCount))
    print('the test  error ratio is %f' %(errorCount/mTest))


if __name__ == "__main__":
    testvec = img2vector('testDigits/0_13.txt')
    print(testvec[0, :63])
    handwritingClassTest()