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
	plt.title('Marginal distribution of p(x1)')

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
	plt.title('Marginal distribution of p(x2)')

	plt.show()

def generateReportC():
	mu1 = np.matrix([[0], [0]])
	mu2 = np.matrix([5])

	sigma11 = np.matrix([[6, 8],[8, 13]])
	sigma12 = np.matrix([[5], [11]])
	sigma22 = np.matrix([14])
	
	X1 = np.linspace(-50.0, 10.0, num=1001)
	X2 = np.linspace(-50.0, 10.0, num=1001)

	cond_sigma = sigma11 - sigma12 * np.linalg.inv(sigma22) * sigma12.transpose()
	normDensity = []
	for x1 in X1:
		xNorms = []
		for x2 in X2:
			cond_mu = mu1 + sigma12 * np.linalg.inv(sigma22) * (x2 - mu2)
			xNorms.append((generateNormalDensity(np.matrix([[x1], [x2]]), cond_mu, cond_sigma)).item(0))	
		normDensity.append(xNorms)

	figure = plt.figure()
	plot = figure.add_subplot(111, projection='3d')
	plot.plot_surface(X1, X2, normDensity, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	plt.show()

def generateReportD():
	mu1 = np.matrix([[0], [0]])
	mu2 = np.matrix([5])

	sigma11 = np.matrix([[6, 8],[8, 13]])
	sigma12 = np.matrix([[5], [11]])
	sigma22 = np.matrix([14])
	
	X1 = np.linspace(-10.0, 20.0, num=101)

	cond_sigma = sigma22 - sigma12.transpose() * np.linalg.inv(sigma11) * sigma12
	normDensity = []
	for x1 in X1:
		cond_mu = mu2 + sigma12.transpose() * np.linalg.inv(sigma11) * (x1 - mu1)
		normDensity.append((generateNormalDensity(np.matrix([x1]), cond_mu, cond_sigma)).item(0))			
			
	figure = plt.figure()

	normDensityPlot, = plt.plot(X1, normDensity, 'r')
	print X1[normDensity.index(max(normDensity))]
	plt.show()

# generateReportA()
# generateReportB()
# generateReportC()
generateReportD()
