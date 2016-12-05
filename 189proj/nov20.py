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
	from sklearn.decomposition import KernelPCA
	plt.figure(1)
	plt.style.use('ggplot')
	kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)

	feature1, val1 = x1**2, valx1**2
	feature2, val2 = x2**0.5, valx2**0.5
	feature3, val3 = x3, valx3
	feature4, val4 = np.mean(x2) * x1 / x2 / x3, np.mean(valx2) * valx1 / valx2 / valx3
	feature5, val5 = np.log(x1**2 / x3), np.log(valx1**2 / valx3)

	X = [[feature1.item(i), feature2.item(i), feature3.item(i), feature4.item(i), feature5.item(i)] for i in range(len(x1))]
	# X_kpca = kpca.fit_transform(X)
	# print X_kpca
	# print np.matrix(X).shape, np.matrix(X_kpca).shape
	from sklearn import svm
	clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(X, score)

	Val = [[val1.item(i), val2.item(i), val3.item(i), val4.item(i), val5.item(i)] for i in range(len(valx1))]
	count = 0
	for i in range(len(Val)):
		if clf.predict(Val[i]).item(0) == valScore[i]:
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
print score_train[:5]

plotGen2D(bet_train[0], bet_train[1], bet_train[2], \
	score_train, bet_val[0], bet_val[1], bet_val[2], score_val)
