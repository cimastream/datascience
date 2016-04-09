#-*- coding: utf-8 -*-
# 이 프로젝트의 목표는, 그룹의 특성을 나타내는 단어를 Naive Bayes를 활용하여 파악하여 학습시킨 후,
# 테스트까지 가능하게 하는 것입니다.


# 수치해석용 Python 라이브러리인 numpy를 호출합니다.
from numpy import *


# 이 함수(loadDataSet)는 클래스를 미리 지정해둔 문장들의 집합을 불러들입니다.
# 문장에 대해서만 클래스를 지정하고, 단어에 대해서 별도로 클래스를 지정하지는 않았습니다.
# 무례하다고 판단된 문장과 그 문장들에서 자주 등장하는 단어 중 어떤 단어가 영향을 주었는지 학습시킬 준비를 합니다.
def loadDataSet():
  WordsList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],   
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
               ['The', 'bag', 'is', 'great', 'very', 'useful'],
               ['This', 'is', 'worthless', 'who', 'does', 'use', 'it', '?']]
  classVec = [0, 1, 0, 1, 0, 1, 0, 1]           # 부정적으로 평가받은 경우 1, 그렇지 않을 경우 0를 부여합니다.
  return WordsList, classVec


# 이 함수는 dataSet을 입력하면 사용된 단어를 하나의 집합으로 합친 후 출력합니다. 
def createVocabList(dataSet):    
  vocabSet = set([])                      # 먼저, 비어있는 집합 vocabSet을 생성합니다.
  for document in dataSet:                # 입력한 dataset의 각 document에 대해서
    vocabSet = vocabSet | set(document)   # document의 집합과 기존 vocabSet의 합집합을 생성합니다.
  return list(vocabSet)                   # 생성된 새 집합 vocabSet의 원소값을 list로 반환합니다.
  

# 이 함수는 입력된 문서의 단어(inputSet)와 기존에 설정된 단어들(vocabList)을 비교한 후, 
# 특정 단어가 사용되었을 경우 vocabList상 해당단어의 위치에 1을 출력합니다.
def setOfWords2Vec(vocabList, inputSet):
  returnVec = [0]*len(vocabList)            # 기존에 설정된 단어들(vocabList)의 개수와 동일한 차원의 영벡터를 생성합니다.
  for word in inputSet:                     # 입력된 문서의 단어들에 대하여  
    if word in vocabList:                   # 각 단어가 기존 설정된 단어들 리스트 vocabList에 존재할 경우
      returnVec[vocabList.index(word)] = 1  # 해당 위치에 1을 출력합니다.
    else:                                   # 그렇지 않을 경우,
      print "the word: %s is not in my Vocabulary!" % word    # 이 문장을 출력합니다.
    return returnVec                        # 완성된 returnVec을 반환합니다.


# Training 함수를 정의합니다. 인자로 받는 값은 loadDataSet 함수에서 반환받은 값들입니다. 
def trainNB0(trainMatrix, trainCategory):
  numTrainDocs = len(trainMatrix)                      # 학습된 데이터 행렬의 문서의 개수를 받습니다.
  numWords = len(trainMatrix[0])                       # 위 데이터 행렬의 0번째 리스트의 개수를 받습니다.
  pAbusive = sum(trainCategory) / float(numTrainDocs)  # 해당 문서에서 Class 1에 해당하는 문서의 비중을 수치로 받습니다.
  p0Num = ones(numWords); p1Num = ones(numWords)       # 문서를 클래스별로 나눌 준비를 하기 위해 
  p0Denom = 2.0; p1Denom = 2.0                         # 변수들의 초기값을 설정합니다.
  for i in range(numTrainDocs):                        # 학습될 문서들을 순서대로
    if trainCategory[i] == 1:                          # 만일 Class 1에 해당되는 문서라면
      p1Num += trainMatrix[i]                          # p1Num에 해당 문서의 벡터형태 리스트를 더하며
      p1Denom += sum(trainMatrix[i])                   # p1Denom에 해당 문서의 벡터값 합을 더합니다.
    else:                                              # 아니라면(만일 Class 0에 해당되는 문서라면)
      p0Num += trainMatrix[i]                          # p0Num에 해당 문서의 벡터형태 리스트를 더하며
      p0Denom += sum(trainMatrix[i])                   # p0Denom에 해당 문서의 벡터값 합을 더합니다.
  p1Vect = p1Num / p1Denom                             # Class 1에 해당되는 문서들의 정규화된 값을 계산합니다.
  p0Vect = p0Num / p0Denom                             # Class 0에 해당되는 문서들의 정규화된 값을 계산합니다.
  return p0Vect, p1Vect, pAbusive                      # 각 값을 반환받습니다.


# 분류함수를 정의합니다. 4개의 인자 중 3개는 위의 trainNB0에서 반환받은 값들입니다.
# vec2Classify 인자는 보통 setOfWords2Vec 함수를 통해 반환받은 리스트입니다.
def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):    
  p1 = sum(vec2Classify * p1Vect) + log(pClass1)          # vec2Classify의 출력값과 p1Vect 값을 곱한후 합하여
  p0 = sum(vec2Classify * p0Vect) + log(1.0 - pClass1)    # pAbusive값(Class1에 해당되는 문서의 비중)에 자연로그를 취한 후 더하여 
  if p1 > p0:                                             # p1값을 계산합니다. 비슷한 방법으로 p0값을 계산하여
    return 1                                              # p1이 더 클 경우는 Class 1을, p0이 더 클 경우는 Class 0을 반환합니다.
  else:                                                   # vec2Classify는 0과 1로 출력된 리스트이고, 
    return 0                                              # p1Vect은 단어의 가중치로 출력된 리스트입니다.


#
def testingNB():
  listOPosts, listClasses = loadDataSet()                                 # loadDataSet 함수의 리턴값을 받습니다.
  myVocabList = createVocabList(listOPosts)                               # createVocabList 함수의 리턴값을 받습니다.
  trainMat = []                                                           # 학습시킬 행렬을 생성하기 위해 빈 리스트로 시작합니다.
  for postingDoc in listOPosts:                                           # listOPost 리스트의 각 문장에 대해서
    trainMat.append(setOfWords2Vec(myVocabList, postingDoc))              # setOFwords2Vec로 0과 1로 표현된 행렬을 trainMat에 추가합니다.
  p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))           # trainNBO의 리턴값을 받습니다.
  testEntry = ['love', 'my', 'dalmation']                                 # 예문의 리스트를(Class 0으로 예측될만한)
  thisDoc = array(setOfWords2Vec(myVocabList, testEntry))                 # 이와 같이 입력하면
  print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)  # 학습된 데이터를 바탕으로 단어의 클래스를 예측합니다.
  testEntry = ['stupid', 'garbage']                                       # 예문의 리스트를(Class 1로 예측될만한)
  thisDoc = array(setOfWords2Vec(myVocabList, testEntry))                 # 이와 같이 입력하면
  print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)  # 학습된 데이터를 바탕으로 단어의 클래스를 예측합니다.


# 앞서 사용된 bagOfWords2Vec의 변형된 함수로써, 단어가 중복사용되어도 값이 매번 반영됩니다.
def bagOfWords2VecMN(vocabList, inputSet):              # inputSet을 기존에서 설정된 vocabList와 비교하여
  returnVec = [0]*len(vocabList)                        # 먼저 vacabList의 단어 리스트를 영벡터 리스트로 변환한 후
  for word in inputSet:                                 # inputSet에 있는 단어가
    if word in vocabList:                               # vocabList에도 있다면
      returnVec[vocabList.index(word)] += 1             # 해당 위치의 값을 1 증가시킵니다.(중복 허용)
      return returnVec                                  # 완성된 값을 반환합니다.


# 참고문헌: Peter Harrington의 저서 <Machine Learning in Action> 외
# 감사합니다.