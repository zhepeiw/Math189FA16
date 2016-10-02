import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def generateNormalDensity(X, mu, cov):
	n = X.shape[0]
	diff = X - mu
	diff = np.matrix(diff)
	exponent = -diff.transpose() * np.linalg.inv(cov) * diff / 2.0
	return np.exp(exponent) / (math.sqrt(2 * math.pi)**n * math.sqrt(np.linalg.det(cov)))

def generateReportA():
	mu1 = [[0], [0]]
	mu1 = np.matrix(mu1)
	sigma1 = [[6,8],[8,13]]
	sigma1 = np.matrix(sigma1)

	X1 = np.linspace(-10.0, 10.0, num=1001)
	X2 = np.linspace(-10.0, 10.0, num=1001)
	normDensitylist = []
	for x1 in X1:
		x1NormList = []
		for x2 in X2:
			X = np.matrix([[x1], [x2]])
			x1NormList.append((generateNormalDensity(X, mu1, sigma1)).item(0))
		normDensitylist.append(x1NormList)
	# print normDensitylist	
	# X = np.matrix([[3], [-3]])
	# print generateNormalDensity(X, mu1, sigma1)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X1, X2, normDensitylist,cmap=cm.coolwarm,linewidth=0, antialiased=False)

	plt.show()

def generateReportB():
	mu2 = [5]
	mu2 = np.matrix(mu2)
	sigma2 = [14]
	sigma2 = np.matrix(sigma2)

	X1 = np.linspace(-10.0, 20.0, num=101)
	normDensitylist = []
	for x1 in X1:
		X = np.matrix([x1])
		normDensitylist.append((generateNormalDensity(X, mu2, sigma2)).item(0))
	# print normDensitylist	
	# X = np.matrix([[3], [-3]])
	# print generateNormalDensity(X, mu1, sigma1)

	fig = plt.figure()
	plt.style.use('ggplot')
	normDensityPlot, = plt.plot(X1, normDensitylist, 'r')

	plt.show()

# generateReportA()
# generateReportB()
