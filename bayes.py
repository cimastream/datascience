#-*- coding: utf-8 -*-
# This is the very beginning of this project.
# The goal of this project is to classify the set of words with probability theory, Naive Bayes.
# I'm studying with the book written by Peter Harrington.

from numpy import *

# 이 함수는 데이터를 읽어들인 후, 
def loadDataSet():
  WordsList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],   # This is an example. I can change these words to others.
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
  classVec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not
  return WordsList, classVec

def createVocabList(dataSet):    # When input a dataset.
  vocabSet = set([])       # Create an empty set
  for document in dataSet:
    vocabSet = vocabSet | set(document)   # Create the union of two sets
  return list(vocabSet)
  
def setOfWords2Vec(vocabList, inputSet):    # When you import this fuction, setOfWords2Vec, with two input variables.
  returnVec = [0]*len(vocabList)    # Create a vector of all 0s
  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)] = 1
    else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

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

