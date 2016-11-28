import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from collections import defaultdict

def plotGen2D(x1, x2, score):
	plt.figure(1)
	plt.style.use('bmh')
	# plt.subplot(211)
	xLose = [np.log(x1[i]) for i in range(len(x1)) if score[i] == -1]
	yLose = [x2[i] for i in range(len(x2)) if score[i] == -1]

	xDraw = [np.log(x1[i]) for i in range(len(x1)) if score[i] == 0]
	yDraw = [x2[i] for i in range(len(x2)) if score[i] == 0]

	xWin = [np.log(x1[i]) for i in range(len(x1)) if score[i] == 1]
	yWin = [x2[i] for i in range(len(x2)) if score[i] == 1]

	losePlot, = plt.plot(xLose, yLose,'ro')
	drawPlot, = plt.plot(xDraw, yDraw,'bo')
	winPlot, = plt.plot(xWin, yWin, 'go')
	plt.legend((losePlot,drawPlot,winPlot), ("Home Lose","Draw","Home Win"), loc=1)

	plt.show()

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

def generateSVM(x1, x2, x3, score, valx1, valx2, valx3, valScore):
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
	return 1.0 * count / len(valx1)

def generateData(trainRatio):
	import parse
	xData, yData = parse.generate()
	xData, yData = np.matrix(xData), np.matrix(yData)
	# print len(yData.tolist()[0])
	trainBound = int(trainRatio * xData.shape[1])
	xTrain, yTrain = xData[:, :trainBound], yData[:, :trainBound]
	xVal, yVal = xData[:, trainBound:], yData[:, trainBound:]
	return xTrain.tolist(), yTrain.tolist()[0], xVal.tolist(), yVal.tolist()[0]

xTrain, yTrain, xVal, yVal = generateData(0.8)
xTrain1 = np.asarray(xTrain[0]).reshape(len(xTrain[0]), 1)
xTrain2 = np.asarray(xTrain[1]).reshape(len(xTrain[1]), 1)
xTrain3 = np.asarray(xTrain[2]).reshape(len(xTrain[2]), 1)
yTrain = np.asarray(yTrain).reshape(len(yTrain), 1)
xVal1 = np.asarray(xVal[0]).reshape(len(xVal[0]), 1)
xVal2 = np.asarray(xVal[1]).reshape(len(xVal[1]), 1)
xVal3 = np.asarray(xVal[2]).reshape(len(xVal[2]), 1)
yVal = np.asarray(yVal).reshape(len(yVal), 1)
del xTrain, xVal


accu = generateSVM(xTrain1, xTrain2, xTrain3, yTrain, \
	xVal1, xVal2, xVal3, yVal)
print "Prediction accuracy: {}".format(str(accu))
# plotGen2D(xTrain1, xTrain3, yTrain)
# Accuracy: 0.491675757719
