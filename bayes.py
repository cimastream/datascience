#-*- coding: utf-8 -*-
# This is the very beginning of this project.
# The goal of this project is to classify the set of words with probability theory, Naive Bayes.
# I'm studying with the book written by Peter Harrington.

from numpy import *

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
    vocabSet = vacabSet | set(document)   # Create the union of two sets
  return list(vocabSet)
  
def setOfWords2Vec(vocabList, inputSet):    # When you import this fuction, setOfWords2Vec, with two input variables.
  returnVec = [0]*len(vocabList)    # Create a vector of all 0s
  for word in inputSet:
    if word in vocabList:
      returnVecv[vocabList.index(word)] = 1
      else: print "the word: %s is not in my Vocabulary!" % word
      return returnVec
      
  
  
  
