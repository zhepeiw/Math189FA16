import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from collections import defaultdict

# df_train = pd.read_csv('E0.csv', sep=',', engine='python')
# df_train.describe()

# df_val = pd.read_csv('E2017.csv', sep=',', engine='python')
# df_val.describe()

# companyList = [
# 	['B365H','B365D','B365A'],
# 	['BWH','BWD','BWA'],
# 	['IWH','IWD','IWA'],
# 	['LBH','LBD','LBA'],
# 	['PSH','PSD','PSA'],
# 	['WHH','WHD','WHA'],
# 	['VCH','VCD','VCA'],
# 	['PSCH','PSCD','PSCA']
# ]

# def generateScoreData(dataSet):
# 	FTHomeScore = dataSet[['FTHG']].as_matrix()
# 	FTAwayScore = dataSet[['FTAG']].as_matrix()

# 	FTScoreDiff = [FTHomeScore.item(i) - FTAwayScore.item(i) for i in range(len(FTHomeScore))]
# 	FTResult = []
# 	for s in FTScoreDiff:
# 		if s > 0:
# 			FTResult.append(1)
# 		elif s < 0:
# 			FTResult.append(-1)
# 		else:
# 			FTResult.append(0)

# 	return FTScoreDiff, FTResult



# def generateBetData(dataSet, betH, betD, betA):
# 	BH = dataSet[[betH]].as_matrix()
# 	BD = dataSet[[betD]].as_matrix()
# 	BA = dataSet[[betA]].as_matrix()

# 	return BH, BD, BA


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

def generateReport(x1, x2, x3, score, valx1, valx2, valx3, valScore):
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
	from sklearn import svm
	clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(X, score)

	count = 0
	for i in range(len(valx1)):
		if clf.predict([val1.item(i), val2.item(i), val3.item(i), val4.item(i), val5.item(i)]).item(0) == valScore[i]:
			count += 1
	print 1.0 * count / len(valx1)

def generateData():
	import parse
	xData, yData = parse.generate()
	xData, yData = np.matrix(xData), np.matrix(yData)
	# print len(yData.tolist()[0])
	trainBound = int(0.9 * xData.shape[1])
	xTrain, yTrain = xData[:, :trainBound], yData[:, :trainBound]
	xVal, yVal = xData[:, trainBound:], yData[:, trainBound:]
	return xTrain.tolist(), yTrain.tolist()[0], xVal.tolist(), yVal.tolist()[0]

xTrain, yTrain, xVal, yVal = generateData()
# print len(yTrain)
xTrain1 = np.asarray(xTrain[0]).reshape(len(xTrain[0]), 1)
xTrain2 = np.asarray(xTrain[1]).reshape(len(xTrain[1]), 1)
xTrain3 = np.asarray(xTrain[2]).reshape(len(xTrain[2]), 1)
yTrain = np.asarray(yTrain).reshape(len(yTrain), 1)
xVal1 = np.asarray(xVal[0]).reshape(len(xVal[0]), 1)
xVal2 = np.asarray(xVal[1]).reshape(len(xVal[1]), 1)
xVal3 = np.asarray(xVal[2]).reshape(len(xVal[2]), 1)
yVal = np.asarray(yVal).reshape(len(yVal), 1)
del xTrain, xVal

generateReport(xTrain1, xTrain2, xTrain3, yTrain, \
	xVal1, xVal2, xVal3, yVal)
