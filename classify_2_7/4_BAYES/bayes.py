#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    '''
    创建包含文档中所有不重复的单词的词汇表
    :param dataSet: 所有进行词条切分后的列表
    :return: 词汇表
    '''
    vocabList = set([])
    for document in dataSet:
        vocabList = vocabList | set(document)
    return list(vocabList)

# 贝努力模型(词集模型）set-of-words 不考虑词在文档中出现的次数，只考虑出不出现，相当于假设词是等权重的
def setOfWords2Vec(vocabList, document):
    '''
    若某个词出现在词汇表中，则将句向量的对应位置置为1
    :param vocabList:词汇表 
    :param inputSet: 某个进行切分词条后的文档
    :return: 
    '''
    returnVec = [0] * len(vocabList)
    # print(returnVec, type(returnVec))
    for word in document:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word %s is not in my Vocabulary! ' % word)
    return returnVec


# 多项式模型（词袋模型）bag-of-words
def bagOfWords2VecMN(vocabList, document):
    returnVec = [0]*len(vocabList)
    for word in document:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 从词向量计算概率
def trainNB0(trainMatrix, trainCategory):
    '''
    计算概率 P(ci|w) = [P(w|ci)P(ci)] / P(w)
    :param trainMatrix: 
    :param trainCategory: 
    :return: 
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = float(sum(trainCategory)) / numTrainDocs

    p1Num = np.ones(numWords)
    p0Num = np.ones(numWords)
    p1Demon = 2.0
    p0Demon = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Demon += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Demon += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Demon)
    p0Vect = np.log(p0Num/p0Demon)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vect, p1Vect, pAbusive):
    p1 = sum(vec2Classify * p1Vect) + np.log(pAbusive)
    # print(p1)
    p0 = sum(vec2Classify * p0Vect) + np.log(1-pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    dataSet, classLabel = loadDataSet()
    myVocabList = createVocabList(dataSet)
    dataMatrix = []
    for document in dataSet:
        dataMatrix.append(setOfWords2Vec(myVocabList, document))
    p0Vect, p1Vect, pAbusive = trainNB0(dataMatrix, classLabel)
    # print(type(p1Vect), p1Vect)
    testEntry = ['I', 'love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'is classified as ', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))
    testEntry = ['stupid', 'dog']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    # print(type(thisDoc), thisDoc)
    print(testEntry, 'is classified as ', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive))


def textParse(bigString):
    '''
    对文本进行词条切分
    :param bigString:一整个文本
    :return: 进行词条切分后的文本
    '''
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok for tok in listOfTokens if len(tok) > 2]


def spamTest():
    '''
    垃圾邮件分类
    :return: 垃圾邮件类别
    '''
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    # print(trainingSet)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    # print('****')
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1v, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # print('****')
        if classifyNB(np.array(wordVector), p0V, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
            print('real label is', classList[docIndex], 'but the predicted label is', classifyNB(np.array(wordVector), p0V, p1v, pSpam), 'original email is', docList[docIndex])

    print('the error rate is %f' % (float(errorCount) / len(testSet)))


def calcMostFreq(vocabList, fullText):
    '''
    遍历词表中的每个词，统计它在文本中出现的次数，从高到低排序，返回排序最高的n个词
    :param vocabList:词汇表
    :param fullText:文本
    :return:
    '''
    import operator
    freqDict = {}
    for word in vocabList:
        freqDict[word] = fullText.count(word)
    sortedFreqDict = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    # print(type(sortedFreqDict), sortedFreqDict[:10])
    return sortedFreqDict[:3]


def stopWords():
    import re
    wordList = open('stopWords.txt').read() # see http://www.ranks.nl/stopwords
    listOfTokens = re.split(r'\W+', wordList)
    print('read stop word from \'stopWord.txt\':', listOfTokens)
    return [tok.lower() for tok in listOfTokens]


def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    # print(len(feed1['entries']))  # 60
    # print(len(feed0['entries']))  # 10
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    print(minLen)
    for i in range(minLen):
        # 每次只去访问一条RSS源
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append([word.lower() for word in wordList])
        # print(docList)
        fullText.append([word.lower() for word in wordList])
        classList.append(1)   
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append([word.lower() for word in wordList])
        fullText.append([word.lower() for word in wordList])
        classList.append(0)
    print(fullText)
    vocabList = createVocabList(fullText)
    print('\nVocabList is ',vocabList)
    print('\nRemove Stop Word:')
    stopWordList = stopWords()
    for stopWord in stopWordList:
        if stopWord in vocabList:
            vocabList.remove(stopWord)
            print('Removed: ', stopWord)

    # top30Words = calcMostFreq(vocabList, fullText)
    # for pairW in top30Words:
    #     if pairW[0] in vocabList:
    #         vocabList.remove(pairW[0])

    trainingSet = list(range(2*minLen))
    testSet = []

    print('\n\nBegin to create a test set: \ntrainingSet:', trainingSet, '\ntestSet', testSet)
    for i in range(5):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    print('random select 5 sets as the testSet:\ntrainingSet:', trainingSet, '\ntestSet', testSet)
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1v, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # print('****')
        if classifyNB(np.array(wordVector), p0V, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
            print('real label is', classList[docIndex], 'but the predicted label is',
                  classifyNB(np.array(wordVector), p0V, p1v, pSpam), 'original email is', docList[docIndex])

    print('the error rate is %f' % (float(errorCount) / len(testSet)))
    return vocabList, p0V, p1v


def testRSS():
    import feedparser
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    sf = feedparser.parse('https://fedoramagazine.org/feed/')
    vocabList,pSF,pNY = localWords(ny, sf)


def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny, sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -4.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -4.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


if __name__ == "__main__":
    # dataSet, classLabel = loadDataSet()
    # myVocabList = createVocabList(dataSet)
    # print(myVocabList)
    # print(setOfWords2Vec(myVocabList, dataSet[0]))
    # trainMatrix = []
    # for document in dataSet:
    #     trainMatrix.append(setOfWords2Vec(myVocabList, document))
    # p0Vect, p1Vect, pAbusive = trainNB0(trainMatrix, classLabel)
    # print('p0Vect: ', p0Vect)
    # print('p1Vect: ', p1Vect)
    # print('pAbusive: ', pAbusive)

    # testingNB()

    # spamTest()

    import feedparser
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    # print(len(ny['entries']))
    sf = feedparser.parse('https://fedoramagazine.org/feed/')
    # print(len(sf['entries']))  # 101  医药
    vocabList, p0V, p1V = localWords(ny, sf)
    # print('vocabList:', vocabList)
    print('p0V:', p0V)
    print('p1V:', p1V)

    getTopWords(ny, sf)