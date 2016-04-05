#-*- coding: utf-8 -*-
# 이 프로젝트의 목표는, 문장의 폭력성에 영향을 주는 단어를 Naive Bayes를 활용하여 학습시킨 후 테스트까지 가능하게 하는 것입니다.
# Peter Harrington의 저서 <Machine Learning in Action> 외의 문헌을 참고하였습니다.


# 수치해석용 Python 라이브러리인 numpy를 호출합니다.
from numpy import *


# 이 함수(loadDataSet)는 미리 지정해둔 무례한/무례하지 않은 문장들의 집합을 불러들입니다. 중립적인 단어들도 중간에 배치되어있습니다.
# 문장에 대해서만 클래스를 지정하고, 단어에 대해서 별도로 클래스를 지정하지는 않았습니다.
# 폭력적이라고 판단한 문장과 그 문장들에서 자주 등장하는 단어 중 어떤 단어가 영향을 주었는지 학습시킬 준비를 합니다.

def loadDataSet():
  WordsList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],   
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
               ['The', 'bag', 'is', 'great', 'very', 'useful'],
               ['This', 'is', 'worthless', 'who', 'does', 'use', 'it', '?']]
  classVec = [0, 1, 0, 1, 0, 1, 0, 1]           # 무례한 표현일 경우 1, 그렇지 않을 경우 0를 부여합니다.
  return WordsList, classVec


# 이 함수는 dataSet을 입력하면 사용된 단어를 하나의 집합으로 합친 후 출력합니다. 
def createVocabList(dataSet):    
  vocabSet = set([])                      # 먼저, 비어있는 집합 vocabSet을 생성합니다.
  for document in dataSet:                # 입력한 dataset의 각 document에 대해서
    vocabSet = vocabSet | set(document)   # document의 집합과 기존 vocabSet의 합집합을 생성합니다.
  return list(vocabSet)                   # 생성된 새 집합 vocabSet의 원소값을 list로 반환합니다.
  

# 이 함수는 입력된 문서의 단어(inputSet)와 기존에 설정된 단어들(vocabList)을 비교한 후, 
# 특정 단어가 사용되었을 경우 해당단어의 위치에 1을 출력합니다.
def setOfWords2Vec(vocabList, inputSet):
  returnVec = [0]*len(vocabList)            # 기존에 설정된 단어들(vocabList)의 개수와 동일한 차원의 영벡터를 생성합니다.
  for word in inputSet:                     # 입력된 문서의 단어들에 대하여  
    if word in vocabList:                   # 각 단어가 기존 설정된 단어들 리스트 vocabList에 존재할 경우
      returnVec[vocabList.index(word)] = 1  # 해당 위치에 1을 출력합니다.
    else: 
      print "the word: %s is not in my Vocabulary!" % word
    return returnVec                        


# Training 함수를 정의합니다. 인자로 받는 값은  
def trainNB0(trainMatrix, trainCategory):
  numTrainDocs = len(trainMatrix)           # 
  numWords = len(trainMatrix[0])            # 
  pAbusive = sum(trainCategory) / float(numTrainDocs)  # 해당 글이 무례하다고 느껴질 확률식을 설정합니다.
  p0Num = zeros(numWords); p1Num = zeros(numWords)     
  p0Denom = 2.0; p1Denom = 2.0              # 
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


# 
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
  p1 = sum(vec2Classify * p1Vec) + log(pClass1)
  p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
  if p1 > p0:
    return 1
  else:
    return 0


#
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
  testEntry = ['stupid', 'garbage', 'worthless']
  thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
  print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


#
def bagOfWords2VecMN(vocabList, imputSet):
  returnVec = [0]*len(vocabList)
  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)] += 1
      return returnVec