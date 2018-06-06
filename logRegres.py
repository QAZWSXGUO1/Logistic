"""
guoweilin
"""

from numpy import *

def loadDataSet():
	"""
	read dataset like [x1,x2,label]
	"""
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat

def Sigmoid(inX):
	return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, ClassLables):
	"""
	calc gradAscent 
	dataMatIn: dataMat from loadDataSet
	ClassLables : labelMat from loadDataSet
	alpha: steplenth
	
	"""
	dataMatrix = mat(dataMatIn)
	labelMat = mat(ClassLables).transpose()
	m,n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n, 1))
	for k in range(maxCycles):
		h = Sigmoid(dataMatrix * weights)
		error = labelMat - h
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights
