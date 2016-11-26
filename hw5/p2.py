import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('classification.csv', sep=',', engine='python').as_matrix()
X = df[:, :2]
ones = np.ones((X.shape[0], 1))
X = np.matrix(np.hstack((ones, X)))
Y = np.matrix(df[:, 2])
V0 = 100 * np.matrix(np.identity(X.shape[1]))

def sigmoid(x):
	return 1. / (1 + np.exp(-x))
def findOptTheta(X, Y, V, tolerance=1e-3, alpha=0.05, max_iter=1000, freq=100):
	theta = np.random.rand(X.shape[0], 1) * np.mean(X)
	theta = np.matrix(theta)
	Vinv = np.linalg.inv(V)
	itr = 0
	while itr < max_iter:
		mu = sigmoid(theta.transpose() * X)
		mu = np.matrix(mu).transpose()
		grad = X * (mu - Y) - 0.5 * (Vinv + Vinv.transpose()) * theta
		# grad = X * (mu - Y) #- 0.5 * (Vinv + Vinv.transpose()) * theta
		# print X.shape, mu.shape, grad.shape
		# break
		theta = theta - (alpha / X.shape[1]) * grad
		if itr % freq == 0:
			print "iteration " + str(itr) + " with gradient norm " + str(np.linalg.norm(grad))
		itr += 1

	return theta



def hessian(theta, X, V):
	mu = sigmoid(theta.transpose() * X)
	S = np.zeros((X.shape[1], X.shape[1]))
	for i in range(S.shape[0]):
		S.itemset((i, i), mu.item(i) * (1 - mu.item(i)))
	del mu
	S = np.matrix(S)

	Vinv = np.linalg.inv(V)
	return X * S * X.transpose() - 0.5 * (Vinv + Vinv.transpose())


theta_opt = findOptTheta(X.transpose(), Y.transpose(), V0)
hessian_opt = hessian(theta_opt, X.transpose(), V0)
# test
# theta = np.random.rand(X.shape[1], 1) * np.mean(X)
# theta = np.matrix(theta)

def findPosterior(theta, theta_opt, H_opt):
	n = H_opt.shape[0]
	denom = (2 * np.pi) ** (n * 0.5) * np.linalg.det(H_opt) ** 0.5
	numer = np.exp(-0.5 * ((theta - theta_opt).transpose() * H_opt * (theta - theta_opt)).item(0))
	return 1.0 * numer / denom

def generateReport2():
	for i in range(20):
		print i * 1.0 / 10, findPosterior(i * 1.0 / 10 * theta_opt, theta_opt, hessian_opt)


def genDataPlot(X, Y):
	# plot of features
	plt.style.use('bmh')
	feature1, feature2 = X[:, 1], X[:, 2]

	feature1Neg = [feature1.item(i) for i in range(Y.shape[1]) if Y.item(i) == 0]
	feature1Pos = [feature1.item(i) for i in range(Y.shape[1]) if Y.item(i) == 1]
	feature2Neg = [feature2.item(i) for i in range(Y.shape[1]) if Y.item(i) == 0]
	feature2Pos = [feature2.item(i) for i in range(Y.shape[1]) if Y.item(i) == 1]

	negPlot, = plt.plot(feature1Neg, feature2Neg, 'bo')
	posPlot, = plt.plot(feature1Pos, feature2Pos, 'rD')

	plt.show()

generateReport2()

# genPlot(X, Y)
