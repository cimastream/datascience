#-*- coding: utf-8 -*-
# This is the very beginning of this project.
# The goal of this project is to classify the set of words with probability theory, Naive Bayes.
# I'm studying with the book written by Peter Harrington.

from numpy import *


# 이 함수는 미리 지정해둔 폭력적/비폭력적 단어들의 집합을 불러들입니다. 중립적인 단어들도 중간에 배치되어있습니다.
def loadDataSet():
  WordsList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],   # This is an example. I can change these words to others.
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
  classVec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not
  return WordsList, classVec


# 이 함수는 dataset을 입력하면 사용된 단어를 하나의 집합으로 만들어 출력합니다.
def createVocabList(dataSet):    
  vocabSet = set([])       # Create an empty set
  for document in dataSet:
    vocabSet = vocabSet | set(document)   # document의 집합과 기존 vocabSet의 합집합을 생성합니다.
  return list(vocabSet)   # 생선된 새 집합 vocabSet의 list 값을 반환합니다.
  

# 이 함수는 입력된 문서의 단어와 기존에 설정된 단어들을 비교한 후, 특정 단어가 사용되었을 경우 1을 출력합니다. (N차원 벡터값)
def setOfWords2Vec(vocabList, inputSet):    # When you import this fuction, setOfWords2Vec, with two input variables.
  returnVec = [0]*len(vocabList)    # Create a vector of all 0s
  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)] = 1
    else: 
      print "the word: %s is not in my Vocabulary!" % word
    return returnVec

# 
def trainNB0(trainMatrix, trainCategory):
  numTrainDocs = len(trainMatrix)    # Number of the training documents
  numWords = len(trainMatrix[0])    # Number of the words
  pAbusive = sum(trainCategory) / float(numTrainDocs)  #
  p0Num = zeros(numWords); p1Num = zeros(numWords)
  p0Denom = 0.0; p1Denom = 0.0
  for i in range(numTrainDocs):
    if trainCategory[i] == 1:
      p1Num += trainMatrix[i]
      p1Denom += sum(trainMatrix[i])
    else:
      p0Num += trainMatrix[i]
      p0Denom += sum(trainMatrix[i])
  p1Vect = p1Num / p1Denom    #change to log()
  p0Vect = p0Num / p0Denom    #change to log()
  return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
  p1 = sum(vec2Classify * p1Vec) + log(pClass1)
  p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
  if p1 > p0:
    return 1
  else:
    return 0


def testingNB():
  listOPosts, listClasses = loadDataSet()
  myVocabList = createVocabList(listOPosts)
  trainMat = []
  for postingDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    pass


def bagOfWords2VecMN(vocabList, imputSet):
  returnVec = [0]*len(vocabList)
  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)] += 1
      return returnVec

