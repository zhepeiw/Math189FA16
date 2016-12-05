import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from time import time
from collections import defaultdict

df_train = pd.read_csv('E0.csv', sep=',', engine='python')
df_train.describe()

df_val = pd.read_csv('E2017.csv', sep=',', engine='python')
df_val.describe()

order = 3

def generatePrediction(x_train, y_train, x_val):
	param = np.polyfit(x_train, y_train, order)
	yGen = np.poly1d(param)

	y_pred = [yGen(x) for x in x_val]
	return y_pred

def plotGen(x_train,y_train, x_val, y_val):
	plt.figure(1)
	plt.style.use('ggplot')

	xSpace = np.linspace(0,max(x_train),100)

	start = time()
	param = np.polyfit(x_train, y_train, order)
	end = time()

	print "Peformance: {}".format(end - start)

	yGen = np.poly1d(param)
	ySpace = [yGen(x) for x in xSpace]

	plt.subplot(211)
	scoreDotPlot, = plt.plot(x_train, y_train,'o')
	linePlot, = plt.plot(xSpace, ySpace, 'b')
	plt.title("Asjusted Training Data & Polynomial Fit")
	plt.xlabel("Adjusted bet data")
	plt.ylabel("Game result")
	plt.legend((scoreDotPlot, linePlot), ('Training data', 'Fitted curve'), loc=3)

	plt.subplot(212)
	y_pred = generatePrediction(x_train, y_train, x_val)		
	y_diff = [y_pred[i] - y_val[i] for i in range(len(y_pred))]
	plt.hist(y_diff, bins=100)
	plt.xlim(-6, 6)
	plt.title("Prediction Error")
	plt.ylabel("Frequency")
	plt.savefig('linreg.pdf', format='pdf')

	correct = 0
	for i in range(len(y_pred)):
		if (y_pred[i] > 0.5 and y_val[i] > 0) \
		or (y_pred[i] < -0.5 and y_val[i] < 0) \
		or (-0.5 <= y_pred[i] and y_pred[i] <= 0.5 and y_val[i] == 0):
			correct += 1
	print "prediction rate", float(correct) / len(y_pred)

	plt.show()

def parseXData(xData):
	muD = np.mean(xData[1])
	return [(xData[0][i] / xData[2][i] + xData[0][i] / xData[1][i] + xData[1][i] / xData[2][i]) / 3 for i in range(len(xData[0]))]

def generateData(trainRatio):
	import parse
	xData, yData = parse.generate(is_score=True)
	xData, yData = np.matrix(xData), np.matrix(yData)

	trainBound = int(trainRatio * xData.shape[1])
	xTrain, yTrain = xData[:, :trainBound], yData[:, :trainBound]
	xVal, yVal = xData[:, trainBound:], yData[:, trainBound:]
	return parseXData(xTrain.tolist()), yTrain.tolist()[0], parseXData(xVal.tolist()), yVal.tolist()[0]


X_train, Y_train, X_val, Y_val = generateData(0.8)
plotGen(X_train, Y_train, X_val, Y_val)

