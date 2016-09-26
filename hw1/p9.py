# =============problem 9================
# Jiawen Zhu, Hemeng Li and Zhenghan Zhang helped me with this problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict

df = pd.read_csv('/Users/FernandoWang/Downloads/iris.csv', sep=',', engine='python')

df.describe()

X = df[['Sepal.Length','Petal.Width']].as_matrix()
y = df.Species.as_matrix()

def generateNormalDensity(X, mu, cov):
	n = X.shape[0]
	diff = X - mu
	diff = np.matrix(diff).transpose()
	exponent = -diff.transpose() * np.linalg.inv(cov) * diff / 2.0
	return np.exp(exponent) / (math.sqrt(2 * math.pi)**n * math.sqrt(np.linalg.det(cov)))

def analysis(X, y, isLinear, reg):
	labels = np.unique(y)
	mu = {}
	p = {}
	cov = {}
	for label in labels:
		p[label] = (y == label).mean()
		mu[label] = X[y == label].mean(axis=0)
		diff = X[y == label] - mu[label]
		diff = np.matrix(diff)
		cov[label] = diff.transpose() * diff / (y == label).sum()
		# print cov
	if isLinear:
		cov = sum((y == label).sum() * cov[label] for label in labels)
		cov = cov / y.shape[0]
		cov = reg * np.diag(np.diag(cov)) + (1 - reg) * cov

	return p, mu, cov



def predict_proba(X, pi, mu, cov):
	prob = np.zeros((X.shape[0], len(pi)))
	if type(cov) is not dict:
		covariance = cov
		cov = defaultdict(lambda: covariance)
		# print cov
	for i, x in enumerate(X):
		for j in range(len(pi)):
			if j == 0:
				label = 'setosa'
			elif j == 1:
				label = 'versicolor'
			else:
				label = 'virginica'
			prob[i,j] = pi[label] * generateNormalDensity(x, mu[label], cov[label])
	prob = prob / prob.sum(axis=1)[:,np.newaxis]
	# print 'prob', prob
	return prob

def generateReport():
	pi, mu, cov = analysis(X, y,False, 0.0)
	yNum = []
	for i in range(len(y)):
		if y[i] == 'setosa':
			yNum.append(0)
		elif y[i] == 'versicolor':
			yNum.append(1)
		else:
			yNum.append(2)
	print('[linear=True, reg=0.00] accuracy={:0.4f}'.format((np.argmax(predict_proba(X, pi, mu, cov), axis=1) == yNum).mean()))
	for reg in np.linspace(0.0, 1.0, 20):
		pi, mu, cov = analysis(X, y, True, reg)
		yNum = []
		for i in range(len(y)):
			if y[i] == 'setosa':
				yNum.append(0)
			elif y[i] == 'versicolor':
				yNum.append(1)
			else:
				yNum.append(2)
		print('[linear=False,reg={:0.2f}] accuracy={:0.4f}'.format(reg, (np.argmax(predict_proba(X, pi, mu, cov), axis=1) == yNum).mean()))
		# print np.argmax(predict_proba(X, pi, mu, cov), axis=1)
		# break

def plotGen():
	PetalWidth = []
	SepalLength = []
	N = X.shape[0]
	for i in range(X.shape[0]):
		PetalWidth.append(X[i][1])
		SepalLength.append(X[i][0])

	plt.style.use('ggplot')
	setosaPlot, = plt.plot(SepalLength[0: int(N / 3)], PetalWidth[0: int(N / 3)], '^')
	plt.setp(setosaPlot, color='yellow')
	
	versicolorPlot, = plt.plot(SepalLength[int(N / 3): int(2.0 * N / 3)], PetalWidth[N / 3: int(2.0 * N / 3)], 'bo')
	plt.setp(versicolorPlot, color='blue')

	virginicaPlot, = plt.plot(SepalLength[int(2.0 * N / 3): ], PetalWidth[int(2.0 * N / 3): ], 'D')
	plt.setp(virginicaPlot, color='red')

	plt.legend((setosaPlot,versicolorPlot,virginicaPlot), ("Setosa","versicolor","virginicaPlot"), loc=2)
	plt.title('Petal Width vs Sepal Length')
	plt.xlabel('Sepal Length')
	plt.ylabel('Petal Width')
	plt.show()

generateReport()
# plotGen()
