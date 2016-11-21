import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

	return FTScoreDiff, FTResult



def generateBetData(dataSet, betH, betD, betA):
	BH = dataSet[[betH]].as_matrix()
	BD = dataSet[[betD]].as_matrix()
	BA = dataSet[[betA]].as_matrix()

	return BH, BD, BA


def plotGen(x1, x2, x3, score):
	plt.figure(1)
	plt.style.use('ggplot')
	# plt.subplot(211)
	xLose = [x1[i] for i in range(len(x1)) if score[i] == -1]
	yLose = [x2[i] for i in range(len(x2)) if score[i] == -1]
	zLose = [x3[i] for i in range(len(x3)) if score[i] == -1]

	xDraw = [x1[i] for i in range(len(x1)) if score[i] == 0]
	yDraw = [x2[i] for i in range(len(x2)) if score[i] == 0]
	zDraw = [x3[i] for i in range(len(x3)) if score[i] == 0]

	xWin = [x1[i] for i in range(len(x1)) if score[i] == 1]
	yWin = [x2[i] for i in range(len(x2)) if score[i] == 1]
	zWin = [x3[i] for i in range(len(x3)) if score[i] == 1]

	# losePlot, = plt.plot(xLose, yLose,'ro')
	# drawPlot, = plt.plot(xDraw, yDraw,'bo')
	# winPlot, = plt.plot(xWin, yWin, 'go')
	# plt.legend((losePlot,drawPlot,winPlot), ("Home Lose","Draw","Home Win"), loc=1)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xLose, yLose, zLose,c='r', marker='o')
	ax.scatter(xDraw, yDraw, zDraw,c='b', marker='o')
	ax.scatter(xWin, yWin, zWin,c='g', marker='o')
	plt.title('Plot')

	plt.show()

def plotGen2D(x1, x2, x3, score, valx1, valx2, valx3, valScore):
	plt.figure(1)
	plt.style.use('ggplot')
	# plt.subplot(211)
	# xLose = [x1[i] for i in range(len(x1)) if score[i] == -1]
	# yLose = [x2[i] for i in range(len(x2)) if score[i] == -1]

	# xDraw = [x1[i] for i in range(len(x1)) if score[i] == 0]
	# yDraw = [x2[i] for i in range(len(x2)) if score[i] == 0]

	# xWin = [x1[i] for i in range(len(x1)) if score[i] == 1]
	# yWin = [x2[i] for i in range(len(x2)) if score[i] == 1]

	X = [[np.log(x1.item(i)), np.log(x2.item(i)), np.log(x3.item(i)), np.log(x1.item(i) / x3.item(i))] for i in range(len(x1))]
	from sklearn import svm
	clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(X, score)

	count = 0
	for i in range(len(valx1)):
		if clf.predict([np.log(valx1.item(i)), np.log(valx2.item(i)), np.log(valx3.item(i)), np.log(valx1.item(i) / valx3.item(i))]).item(0) == valScore[i]:
			count += 1
	print 1.0 * count / len(valx1)


	# losePlot, = plt.plot(xLose, yLose,'rD')
	# drawPlot, = plt.plot(xDraw, yDraw,'bD')
	# winPlot, = plt.plot(xWin, yWin, 'gD')
	# plt.legend((losePlot,drawPlot,winPlot), ("Home Lose","Draw","Home Win"), loc=1)

	# plt.show()


score_train = generateScoreData(df_train)[1]
# print len(Y_train)
bet_train = generateBetData(df_train, companyList[0][0], companyList[0][1],companyList[0][2])

score_val = generateScoreData(df_val)[1]
# print len(Y_train)
bet_val = generateBetData(df_val, companyList[0][0], companyList[0][1],companyList[0][2])
# print X_train.shape

plotGen2D(bet_train[0], bet_train[1], bet_train[2], \
	score_train, bet_val[0], bet_val[1], bet_train[2], score_val)
