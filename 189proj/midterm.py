import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict

df_train = pd.read_csv('E0.csv', sep=',', engine='python')
df_train.describe()

df_val = pd.read_csv('E2017.csv', sep=',', engine='python')
df_val.describe()

companyList = [
	['B365H','B365D','B365A'],
	['BWH','BWD','BWA'],
	['IWH','IWD','IWA'],
	['LBH','LBD','LBA'],
	['PSH','PSD','PSA'],
	['WHH','WHD','WHA'],
	['VCH','VCD','VCA'],
	['PSCH','PSCD','PSCA']
]



def generateScoreData(dataSet):
	FTHomeScore = dataSet[['FTHG']].as_matrix()
	FTAwayScore = dataSet[['FTAG']].as_matrix()

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



def generateBetData(dataSet, betH, betD, betA):
	BH = dataSet[[betH]].as_matrix()
	BD = dataSet[[betD]].as_matrix()
	BA = dataSet[[betA]].as_matrix()
	# muD = np.mean(BD)
	# betRate = []
	# for i in range(N):
	# 	rate = BH.item(i) / BA.item(i) * muD / BD.item(i)
	# 	betRate.append(rate)

	return BH, BD, BA

def generateXData(dataSet):
	hSum = None
	dSum = None
	aSum = None

	for company in companyList:
		currH, currD, currA = generateBetData(dataSet, company[0], company[1], company[2])
		if hSum is None:
			hSum = np.matrix([0.] * len(currH)).reshape(-1,1)
			dSum = np.matrix([0.] * len(currD)).reshape(-1,1)
			aSum = np.matrix([0.] * len(currA)).reshape(-1,1)
		# print hSum
		hSum += currH
		dSum += currD
		aSum += currA
	
	muD = np.mean(dSum)
	return [hSum.item(i) / aSum.item(i) * muD / dSum.item(i) for i in range(len(hSum))]

# def linreg(X, y, reg=1.0):	
# 	X = np.matrix(X).reshape(-1,1)
# 	y = np.matrix(y).reshape(-1,1)
# 	X = np.hstack((np.ones_like(y), X))
	
# 	eye = np.eye(X.shape[1])
# 	eye[0,0] = 0. # don't regularize bias term!	
# 	return np.linalg.solve(X.transpose() * X + reg * eye, X.transpose() * y)

def generatePrediction(x_train, y_train, x_val, order):
	param = np.polyfit(x_train, y_train, order)
	yGen = np.poly1d(param)
	# paramList = [linregMat.item(i) for i in range(len(linregMat))]

	y_pred = [yGen(x) for x in x_val]
	return y_pred

def plotGen(x_train,y_train, x_val, y_val):
	plt.figure(1)
	plt.style.use('ggplot')
	# plt.subplot(211)
	scoreDotPlot, = plt.plot(x_train, y_train,'o')

	xSpace = np.linspace(0,max(x_train),100)
	param = np.polyfit(x_train, y_train, 3)
	print param
	
	yGen = np.poly1d(param)
	ySpace = [yGen(x) for x in xSpace]
	linePlot, = plt.plot(xSpace, ySpace, 'b')

	# plt.subplot(212)
	# y_pred = generatePrediction(x_train, y_train, x_val, 3)	
	# # print y_pred
	# y_diff = [y_pred[i] - y_val[i] for i in range(len(y_pred))]
	# predDiffPlot, = plt.plot(x_val, y_diff, 'D')

	# correct = 0
	# for i in range(len(y_pred)):
	# 	if (y_pred[i] > 0.5 and y_val[i] > 0) \
	# 	or (y_pred[i] < -0.5 and y_val[i] < 0) \
	# 	or (-0.5 <= y_pred[i] and y_pred[i] <= 0.5 and y_val[i] == 0):
	# 		correct += 1
	# print "prediction rate", float(correct) / len(y_pred)

	

	plt.show()


Y_train = generateScoreData(df_train)
X_train = generateXData(df_train)
Y_val = generateScoreData(df_val)
X_val = generateXData(df_val)

plotGen(X_train, Y_train, X_val, Y_val)
# print linreg(X, Y)
