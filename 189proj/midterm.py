import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict

df = pd.read_csv('E0.csv', sep=',', engine='python')
df.describe()

companyList = [
	['B365H','B365D','B365A'],
	['BWH','BWD','BWA'],
	['IWH','IWD','IWA'],
	['LBH','LBD','LBA'],
	['PSH','PSD','PSA']
]

def generateScoreData():
	FTHomeScore = df[['FTHG']].as_matrix()
	FTAwayScore = df[['FTAG']].as_matrix()

	FTScoreDiff = [FTHomeScore.item(i) - FTAwayScore.item(i) for i in range(len(FTHomeScore))]
	FTResult = []
	for s in FTScoreDiff:
		if s > 0:
			FTResult.append(1)
		elif s < 0:
			FTResult.append(-1)
		else:
			FTResult.append(0)

	return FTScoreDiff



def generateBetData(betH, betD, betA):
	BH = df[[betH]].as_matrix()
	BD = df[[betD]].as_matrix()
	BA = df[[betA]].as_matrix()
	N = len(BH)
	muD = np.mean(BD)
	betRate = []
	for i in range(N):
		rate = BH.item(i) / BA.item(i) * muD / BD.item(i)
		betRate.append(rate)

	return betRate

def generateXData():
	X = []
	for company in companyList:
		X.append(generateBetData(company[0], company[1], company[2]))
	return X

def linreg(X, y, reg=1.0):	
	X = np.matrix(X).transpose()
	y = np.matrix(y).reshape(-1,1)
	X = np.hstack((np.ones_like(y), X))
	
	eye = np.eye(X.shape[1])
	eye[0,0] = 0. # don't regularize bias term!	
	return np.linalg.solve(X.transpose() * X + reg * eye, X.transpose() * y)

def plotGen(x,y):
	plt.style.use('ggplot')
	# scoreDotPlot, = plt.plot(x,y,'o')
	xSpace = np.linspace(0,max(x),100)
	linregMat = linreg(x, y)
	b = linregMat.item(0)

	ySpace = [k*x + b for x in xSpace]
	linePlot, = plt.plot(xSpace, ySpace, 'b')

	plt.show()


Y = generateScoreData()
# X = generateBetData(companyList[1][0], companyList[1][1], companyList[1][2])
X = generateXData()
# plotGen(X,Y)
print linreg(X,Y)
