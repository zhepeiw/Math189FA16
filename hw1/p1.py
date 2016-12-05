# ===========problem 1============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math


# ====part c====
df = pd.read_csv('/Users/FernandoWang/Downloads/online_news_popularity.csv', sep=', ', engine='python')

df['type'] = ''
df.loc[0: int(2.0 / 3 * len(df)), 'type'] = 'train'
df.loc[int(2.0 / 3 * len(df)): int(5.0 / 6 * len(df)), 'type'] = 'validation'
df.loc[int(5.0 / 6 * len(df)): , 'type'] = 'test'
df.describe()

X_train = df[df.type == 'train'][[col for col in df.columns if col not in ['url', 'shares', 'type']]]
y_train = np.log(df[df.type == 'train'].shares).reshape(-1,1)

X_val = df[df.type == 'validation'][[col for col in df.columns if col not in ['url', 'shares', 'type']]]
y_val = np.log(df[df.type == 'validation'].shares).reshape(-1,1)

X_test = df[df.type == 'test'][[col for col in df.columns if col not in ['url', 'shares', 'type']]]
y_test = np.log(df[df.type == 'test'].shares).reshape(-1,1)

X_train = np.hstack((np.ones_like(y_train), X_train))
X_val = np.hstack((np.ones_like(y_val), X_val))
X_test = np.hstack((np.ones_like(y_test), X_test))

def linreg(X, y, reg=0.0):
	eye = np.eye(X.shape[1])
	eye[0,0] = 0. # don't regularize bias term!
	X = np.matrix(X)
	y = np.matrix(y)
	return np.linalg.solve(X.transpose() * X + reg * eye, X.transpose() * y)

# theta_optimal = linreg(X_train, y_train, reg=9.34604374950277)

def generateLambda():
	return [random.uniform(0.0, 150.0) for x in range(150)]

lambList = generateLambda()
thetaList = [linreg(X_train, y_train, reg=la) for la in lambList]

def generateNorm():
	thetaNormList = [np.linalg.norm(x) for x in thetaList]
	return thetaNormList
	# plt.style.use('ggplot')
	# normPlot = plt.plot(lambList, thetaNormList, 'o')
	# plt.setp(normPlot, color='blue')

def generateMse():
	MSE = []
	for i in range(len(lambList)):
		theta = thetaList[i]
		thetaMat = np.matrix(theta)

		s = 0
		for j in range(len(X_val)):
			XvalMat = np.matrix(X_val[j])
			yReg = XvalMat * thetaMat
			s += (y_val.item(j) - yReg.item(0)) ** 2
		s /= len(X_val)
		s = math.sqrt(s)
		MSE.append(s)
	return MSE

def generateMseDict():
	mseDict = {}
	for i in range(len(lambList)):
		theta = thetaList[i]
		thetaMat = np.matrix(theta)

		s = 0
		for j in range(len(X_val)):
			XvalMat = np.matrix(X_val[j])
			yReg = XvalMat * thetaMat
			s += (y_val.item(j) - yReg.item(0)) ** 2
		s /= len(X_val)
		s = math.sqrt(s)
		mseDict[lambList[i]] = s
	return mseDict

def plotGen():
	plt.figure(1)

	# plot mse
	plt.style.use('ggplot')
	# plt.subplot(121)

	mseDict = generateMseDict()
	sortedLambList = [key for key in sorted(mseDict)]
	MSE = [mseDict[k] for k in sortedLambList]

	MSEPlot, = plt.plot(sortedLambList, MSE)
	plt.setp(MSEPlot, color='red')
	plt.title('RMSE vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('RMSE')
	# plt.legend((MSEPlot), ("RMSE"), loc=1)

	# plot norm
	plt.figure(2)
	plt.style.use('ggplot')
	# plt.subplot(122)
	norm = generateNorm()
	normDict = {}
	for i in range(len(norm)):
		normDict[lambList[i]] = norm[i]
	sortedLambList = [key for key in sorted(normDict)]
	norm = [normDict[k] for k in sortedLambList]
	normPlot, = plt.plot(sortedLambList, norm)
	# 
	plt.setp(normPlot, color='blue')
	plt.title('Norm vs Lambda')
	plt.xlabel('lambda')
	plt.ylabel('norm')
	# plt.legend((normPlot), ("norm"), loc=1)

	plt.tight_layout()
	plt.show()

plotGen()

def findOptReg():
	MSE = generateMse()
	minMSE = min(MSE)
	minIndex = MSE.index(minMSE)
	return lambList[minIndex]

# ========part d===========
X_train_d = X_train[:,1:]
X_val_d = X_val[:,1:]
def linregD(X, y):
	eye = np.eye(X.shape[0])
	oneMat = np.ones(X.shape[0])
	regOpt = findOptReg()

	X = np.matrix(X)
	y = np.matrix(y)

	A_mod = X.transpose() * (eye - oneMat / X.shape[0])
	# print A_mod.shape

	theta_opt = np.linalg.solve(A_mod * X + regOpt * np.eye(A_mod.shape[0]), A_mod * y)
	b_opt = sum((y_train - X_train_d * theta_opt)) / X.shape[0]

	original = linreg(X_train, y_train, regOpt)

	b_diff = abs((original[0] - b_opt).item(0))
	theta_diff = np.linalg.norm(theta_opt - original[1:])
	return b_diff, theta_diff
# print linregD(X_train_d, y_train)
# =============part e===============
def plotGenE(OBJ1, OBJ2, ITER):
	# plot mse
	plt.style.use('ggplot')
	# mseDict = generateMseDict()
	# sortedLambList = [key for key in sorted(mseDict)]
	# MSE = [mseDict[k] for k in sortedLambList]

	trainPlot, = plt.plot(ITER, OBJ1)
	plt.setp(trainPlot, color='red')
	normPlot, = plt.plot(ITER, OBJ2)
	plt.setp(normPlot, color='blue')
	plt.legend((trainPlot, normPlot), ('Train', 'Validation'), loc=1)
	plt.title('RMSE vs iteration')
	plt.xlabel('iteration')
	plt.ylabel('RMSE')
	

	plt.show()

def gradientDescent():
	X_train_d = X_train[:,1:]
	X_val_d = X_val[:,1:]

	MSE = generateMse()
	shape = (X_train_d.shape[1], 1)
	n = X_train_d.shape[0]
	epsillon = 1e-6
	ITERNUM = 150
	linreg_theta = 2.5e-12
	linreg_b = 0.2
	regOpt = findOptReg()
	theta_opt = linreg(X_train, y_train, reg=regOpt)
	theta_ = np.zeros(shape)
	b_ = 0

	X_train_dd = np.matrix(X_train_d)
	X_val_dd = np.matrix(X_val_d)

	X_train_d = X_train_dd
	X_val_d = X_val_dd

	grad_theta = np.ones_like(theta_)
	grad_b = np.ones_like(b_)

	obj_train = []
	obj_val = []

	print('==> Training.')
	while np.linalg.norm(grad_theta) > epsillon and \
		np.abs(grad_b) > epsillon and \
		len(obj_train) < ITERNUM:
		trainNorm = np.linalg.norm((X_train_d * theta_).reshape(-1, 1) + b_ - y_train, ) ** 2 / y_train.shape[0]
		obj_train.append(np.sqrt(trainNorm))
		valNorm = np.linalg.norm((X_val_d * theta_).reshape(-1, 1) + b_ - y_val, ) ** 2 / y_val.shape[0]
		obj_val.append(np.sqrt(valNorm))

		grad_theta = ((X_train_d.transpose() * X_train_d + regOpt * np.eye(shape[0])) * theta_ + X_train_d.transpose() * (b_ - y_train)) / X_train_d.shape[0]
		grad_b = ((X_train_d * theta_).sum() - y_train.sum() + b_ * n) / X_train_d.shape[0]

		theta_ = theta_ - linreg_theta * grad_theta
		b_ = b_ - linreg_b * grad_b

		if len(obj_train) % 25 == 0:
			pass
			# print('-- finishing iteration {} - objective {:5.4f} - grad {}'.format(len(obj_train), obj_train[-1], np.linalg.norm(grad_theta)))
	# print np.linalg.norm(theta_opt), b_
	# b_diff = abs((theta_opt[0] - b_).item(0))
	# print b_diff

	print('==> Distance between intercept and orig: {}'.format(abs(theta_opt.item(0) - b_)))
	print('==> Distance between theta and original: {}'.format(np.linalg.norm(theta_ - theta_opt[1:])))

	# plot
	# plotGenE(obj_train, obj_val, range(ITERNUM))
# gradientDescent()

