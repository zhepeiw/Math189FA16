import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from collections import defaultdict

def generateData(trainRatio):
	import parse
	xData, yData = parse.generate()
	xData, yData = np.matrix(xData), np.matrix(yData)
	# print len(yData.tolist()[0])
	trainBound = int(trainRatio * xData.shape[1])
	xTrain, yTrain = xData[:, :trainBound], yData[:, :trainBound]
	xVal, yVal = xData[:, trainBound:], yData[:, trainBound:]
	return xTrain.tolist(), yTrain.tolist()[0], xVal.tolist(), yVal.tolist()[0]

def plotGen2D(x1, x2, score):
	plt.figure(1)
	plt.style.use('ggplot')
	# plt.subplot(211)
	xLose = [x1[i] for i in range(len(x1)) if score[i] == -1]
	yLose = [x2[i] for i in range(len(x2)) if score[i] == -1]

	xDraw = [x1[i] for i in range(len(x1)) if score[i] == 0]
	yDraw = [x2[i] for i in range(len(x2)) if score[i] == 0]

	xWin = [x1[i] for i in range(len(x1)) if score[i] == 1]
	yWin = [x2[i] for i in range(len(x2)) if score[i] == 1]

	plt.subplot(1,3,1)
	winPlot, = plt.plot(xWin, yWin, 'go')
	plt.title('Home Wins')
	plt.axis([0.0, 15.0, 0.0, 15.0])
	plt.subplot(1,3,2)
	drawPlot, = plt.plot(xDraw, yDraw,'bo')
	plt.title('Draws')
	plt.axis([0.0, 15.0, 0.0, 15.0])
	plt.subplot(1,3,3)
	losePlot, = plt.plot(xLose, yLose,'ro')
	plt.title('Home Losses')
	plt.axis([0.0, 15.0, 0.0, 15.0])
	plt.show()

def plotGen3D(x1, x2, x3, score):
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

	ax = plt.subplot(131, projection='3d')
	ax.scatter(xLose, yLose, zLose,c='r', marker='o')
	ax.set_xlim3d(0, 10)
	ax.set_ylim3d(0, 10)
	ax.set_zlim3d(0, 10)
	plt.title('Losses')
	ax = plt.subplot(132, projection='3d')
	ax.scatter(xDraw, yDraw, zDraw,c='b', marker='o')
	ax.set_xlim3d(0, 10)
	ax.set_ylim3d(0, 10)
	ax.set_zlim3d(0, 10)
	plt.title('Draws')
	ax = plt.subplot(133, projection='3d')
	ax.scatter(xWin, yWin, zWin,c='g', marker='o')
	ax.set_xlim3d(0, 10)
	ax.set_ylim3d(0, 10)
	ax.set_zlim3d(0, 10)
	plt.title('Wins')

	plt.show()

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

def generateFeature(x1, x2, x3):
	mD = np.mean(x2)
	return [(x1.item(i) / x3.item(i) + x1.item(i) / x2.item(i) + x2.item(i) / x3.item(i)) / 3 for i in range(x1.shape[0])]

def genReportNaive(xVal1, xVal2, xVal3, yVal):
	count = 0
	for i in range(len(xVal1)):
		yPred = np.argmin([xVal1[i], xVal2[i], xVal3[i]]) - 1
		if yPred == yVal[i]:
			count += 1
	return 1.0 * count / len(xVal1)
	
def genReport1D(fTrain, yTrain, fVal, yVal):
	from scipy.stats import norm
	xLose = [fTrain[i] for i in range(len(fTrain)) if yTrain[i] == -1]
	xDraw = [fTrain[i] for i in range(len(fTrain)) if yTrain[i] == 0]
	xWin = [fTrain[i] for i in range(len(fTrain)) if yTrain[i] == 1]

	mLose, sLose = np.mean(xLose), np.std(xLose)
	mDraw, sDraw = np.mean(xDraw), np.std(xDraw)
	mWin, sWin = np.mean(xWin), np.std(xWin)

	rvLose, rvDraw, rvWin = norm(mLose, sLose), norm(mDraw, sDraw), norm(mWin, sWin)
	count = 0
	for i in range(len(fVal)):
		feature = fVal[i]
		pLose, pDraw, pWin = rvLose.pdf(feature), rvDraw.pdf(feature), rvWin.pdf(feature)
		yPred = np.argmax([pLose, pDraw, pWin]) - 1
		if yPred == yVal[i]:
			count += 1
	return 1.0 * count / len(fVal)

def plotHist(f, score):
	plt.figure(1)
	plt.style.use('ggplot')
	# plt.subplot(211)
	xLose = [f[i] for i in range(len(f)) if score[i] == -1]
	xDraw = [f[i] for i in range(len(f)) if score[i] == 0]
	xWin = [f[i] for i in range(len(f)) if score[i] == 1]

	plt.subplot(1,3,1)
	plt.hist(np.log(xWin), bins=100)
	plt.xlim(-3, 3)
	plt.title('Home Wins')

	plt.subplot(1,3,2)
	plt.hist(np.log(xDraw), bins=100)
	plt.xlim(-3, 3)
	plt.title('Draws')

	plt.subplot(1,3,3)
	plt.hist(np.log(xLose), bins=100)
	plt.xlim(-3, 3)
	plt.title('Home Losses')

	plt.show()

fTrain = generateFeature(xTrain1, xTrain2, xTrain3)
fVal = generateFeature(xVal1, xVal2, xVal3)

# plotHist(fTrain, yTrain)
print "naive: choosing min " , genReportNaive(xVal1, xVal2, xVal3, yVal)
print "1D prediction: ", genReport1D(fTrain, yTrain, fVal, yVal)


# accu = generateSVM(xTrain1, xTrain2, xTrain3, yTrain, \
# 	xVal1, xVal2, xVal3, yVal)
# print "Prediction accuracy: {}".format(str(accu))
# Accuracy: 0.491675757719

# plotGen2D(xTrain1, xTrain3, yTrain)
# plotGen3D(xTrain1, xTrain2, xTrain3, yTrain)
