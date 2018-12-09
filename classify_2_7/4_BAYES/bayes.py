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

# 贝努力模型(词集模型）set-of-words
def setOfWords2Vec(vocabList, document):
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
    vocabList,p0V,p1V=localWords(ny,sf)
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

 #    arr = np.array([-3.04452244 -3.04452244 -3.04452244 -3.04452244 -2.35137526 -2.35137526
 # -3.04452244 -3.04452244 -3.04452244 -3.04452244 -1.65822808 -1.94591015
 # -3.04452244 -2.35137526 -2.35137526 -2.35137526 -2.35137526 -1.94591015
 # -3.04452244 -2.35137526 -2.35137526 -2.35137526 -3.04452244 -2.35137526
 # -2.35137526 -3.04452244 -3.04452244 -3.04452244 -3.04452244 -3.04452244
 # -3.04452244 -2.35137526])
 #    list1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 #    print(type(arr * list1))

    # mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
    # # mySent = mySent.split()
    # # print(mySent)
    # import re
    # regEx = re.compile('\\W+')
    # listOfTokens = regEx.split(mySent)
    # # print(listOfTokens)
    # listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
    #
    # print(listOfTokens)
    #
    # emailText = open('email/ham/6.txt').read()
    # listOfTokens = regEx.split(emailText)
    # print(listOfTokens)

    # spamTest()

    # import feedparser
    # ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    # print(ny)
    # print(ny['entries'])

    # top30Words = [('N', 0), ('b', 0), ('x', 0), ('g', 0), ('a', 0), ('E', 0), ('j', 0), ('c', 0), ('C', 0), ('h', 0)]
    # vocabList = ['N', 'b', 'z']
    # for pairW in top30Words:
    #     if pairW[0] in vocabList:
    #         print(pairW[0])
    #         vocabList.remove(pairW[0])
    # print(vocabList)

    # voc = ['This', 'artist', 'conception', 'the', 'core', 'Cygnus', 'shows', 'the', 'dusty', 'donut', 'shaped', 'surroundings', 'called', 'torus', 'and', 'jets', 'launching', 'from', 'its', 'center', 'Welcome', 'back', 'this', 'series', 'building', 'faster', 'web', 'pages', 'The', 'last', 'article', 'talked', 'about', 'what', 'you', 'can', 'achieve', 'just', 'through', 'image', 'compression', 'The', 'example', 'started', 'with', '2MB', 'browser', 'fat', 'and', 'reduced', 'down', 'weight', '488', '9KB', 'That', '8217', 'still', 'not', 'fast', 'enough', 'This', 'article', 'continues', 'the', 'browser', 'diet', 'lose', 'more', 'fat', 'You', 'might', '8230', 'Viewed', 'from', 'window', 'inside', 'the', 'cupola', 'the', 'International', 'Space', 'Station', 'window', 'the', 'world', 'the', 'Japanese', 'Exploration', 'Agency', 'Transfer', 'Vehicle', 'Linux', 'containers', 'are', 'processes', 'with', 'certain', 'isolation', 'features', 'provided', 'Linux', 'kernel', 'including', 'filesystem', 'process', 'and', 'network', 'isolation', 'Containers', 'help', 'with', 'portability', 'applications', 'can', 'distributed', 'container', 'images', 'along', 'with', 'their', 'dependencies', 'and', 'run', 'virtually', 'any', 'Linux', 'system', 'with', 'container', 'runtime', 'Although', 'container', 'technologies', 'exist', 'for', 'very', 'long', 'time', '8230', 'During', 'National', 'Hispanic', 'Heritage', 'Month', 'celebrating', 'the', 'achievements', 'astronaut', 'Ellen', 'Ochoa', 'and', 'other', 'Hispanic', 'astronauts', 'and', 'professionals', 'NASA', 'Floating', 'upside', 'down', 'and', 'reading', 'checklist', 'may', 'not', 'how', 'most', 'perform', 'the', 'day', 'work', 'but', 'was', 'for', 'Ochoa', 'Space', 'Shuttle', 'Discovery', 'STS', 'mission', 'Fedora', 'delightful', 'use', 'graphical', 'operating', 'system', 'You', 'can', 'point', 'and', 'click', 'your', 'way', 'through', 'just', 'about', 'any', 'task', 'easily', 'But', 'you', '8217', 'probably', 'seen', 'there', 'powerful', 'command', 'line', 'under', 'the', 'hood', 'try', 'out', 'shell', 'just', 'open', 'the', 'Terminal', 'application', 'your', 'Fedora', 'system', 'This', 'article', '8230', 'This', 'view', 'southern', 'California', 'was', 'taken', 'the', 'Apollo', 'crew', 'during', 'their', '18th', 'revolution', 'the', 'Earth', 'Oct', '1968', 'Lots', 'web', 'developers', 'want', 'achieve', 'fast', 'loading', 'web', 'pages', 'more', 'page', 'views', 'come', 'from', 'mobile', 'devices', 'making', 'websites', 'look', 'better', 'smaller', 'screens', 'using', 'responsive', 'design', 'just', 'one', 'side', 'the', 'coin', 'Browser', 'Calories', 'can', 'make', 'the', 'difference', 'loading', 'times', 'which', 'satisfies', 'not', 'just', 'the', 'user', 'but', 'search', 'engines', 'that', '8230', 'Cosmonaut', 'Alexey', 'Ovchinin', 'Roscosmos', 'left', 'and', 'astronaut', 'Nick', 'Hague', 'NASA', 'right', 'embrace', 'their', 'families', 'after', 'landing', 'the', 'Krayniy', 'Airport', 'Some', 'weeks', 'ago', 'Steam', 'announced', 'new', 'addition', 'Steam', 'Play', 'with', 'Linux', 'support', 'for', 'Windows', 'games', 'using', 'Proton', 'fork', 'from', 'WINE', 'This', 'capability', 'still', 'beta', 'and', 'not', 'all', 'games', 'work', 'Here', 'are', 'some', 'more', 'details', 'about', 'Steam', 'and', 'Proton', 'According', 'the', 'Steam', 'website', 'there', 'are', 'new', 'features', 'the', 'beta', 'release', '8230', 'This', 'composite', 'image', 'shows', 'the', 'International', 'Space', 'Station', 'with', 'crew', 'three', 'onboard', 'silhouette', 'transits', 'the', 'Sun', 'roughly', 'five', 'miles', 'per', 'second', 'Sunday', 'Oct', '2018', 'Fedora', 'Classroom', 'sessions', 'continue', 'next', 'week', 'with', 'session', 'Fedora', 'Modularity', 'The', 'general', 'schedule', 'for', 'sessions', 'appears', 'the', 'wiki', 'You', 'can', 'also', 'find', 'resources', 'and', 'recordings', 'from', 'previous', 'sessions', 'there', 'Here', 'are', 'details', 'about', 'this', 'week', 'session', 'Tuesday', 'October', '1400', 'UTC', 'That', 'link', 'allows', 'you', 'convert', 'the', 'time', 'your', 'timezone', 'Topic', 'Fedora', '8230', 'The', 'Soyuz', 'rocket', 'rolled', 'out', 'train', 'the', 'launch', 'pad', 'Tuesday', 'Oct', '2018', 'for', 'the', 'Expedition', 'launch', 'Last', 'month', 'the', 'GNOME', 'project', 'announced', 'the', 'release', 'GNOME', 'The', 'good', 'news', 'that', 'this', 'new', 'version', 'GNOME', 'default', 'the', 'forthcoming', 'release', 'Fedora', 'Workstation', 'GNOME', 'includes', 'range', 'new', 'features', 'and', 'enhancements', 'including', 'improvements', 'Files', 'nautilus', 'and', 'the', 'new', 'Podcasts', 'application', 'The', 'great', 'news', '8230', 'During', 'National', 'Hispanic', 'Heritage', 'Month', 'celebrating', 'the', 'contributions', 'the', 'brilliant', 'Hispanic', 'women', 'and', 'men', 'NASA', 'this', 'image', 'astronaut', 'Joe', 'Acaba', 'installs', 'botany', 'gear', 'for', 'the', 'International', 'Space', 'Station', 'Veggie', 'facility', 'demonstrate', 'plant', 'growth', 'space', 'Swift', 'general', 'purpose', 'programming', 'language', 'built', 'using', 'modern', 'approach', 'safety', 'performance', 'and', 'software', 'design', 'patterns', 'aims', 'the', 'best', 'language', 'for', 'variety', 'programming', 'projects', 'ranging', 'from', 'systems', 'programming', 'desktop', 'applications', 'and', 'scaling', 'cloud', 'services', 'Read', 'more', 'about', 'and', 'how', 'try', 'out', '8230', 'The', 'landing', 'jets', 'fire', 'the', 'Soyuz', 'spacecraft', 'lands', 'with', 'Drew', 'Feustel', 'Ricky', 'Arnold', 'and', 'Oleg', 'Artemyev', 'members', 'the', 'Expedition', 'and', 'crews', 'onboard', 'the', 'International', 'Space', 'Station', 'You', 'may', 'have', 'already', 'seen', 'the', 'article', 'here', 'the', 'Magazine', 'about', 'upscaling', 'bitmap', 'images', 'with', 'better', 'quality', 'That', 'article', 'covered', 'few', 'utilities', 'achieve', 'good', 'results', 'but', 'there', '8217', 'always', 'room', 'for', 'enhancement', 'Meet', 'Waifu2x', 'sophisticated', 'tool', 'that', 'uses', 'deep', 'convolutional', 'neural', 'networks', 'machine', 'learning', 'for', 'short', 'Therefore', 'benefits', 'from', 'trained', '8230', 'International', 'Space', 'Station', 'Commander', 'Alexander', 'Gerst', 'has', 'better', 'view', 'our', 'home', 'planet', 'than', 'most', 'The', 'Linux', 'desktop', 'ecosystem', 'offers', 'multiple', 'window', 'managers', 'WMs', 'Some', 'are', 'developed', 'part', 'desktop', 'environment', 'Others', 'are', 'meant', 'used', 'standalone', 'application', 'This', 'the', 'case', 'tiling', 'WMs', 'which', 'offer', 'more', 'lightweight', 'customized', 'environment', 'This', 'article', 'presents', 'five', 'such', 'tiling', 'WMs', 'for', 'you', 'try', 'out', '8230']
    # print(createVocabList(voc))

    # testRSS()
    import feedparser
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    sf = feedparser.parse('https://fedoramagazine.org/feed/')
    getTopWords(ny, sf)