import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from collections import namedtuple

red = pd.read_csv('./winequality-red.csv', sep=';').as_matrix()
white = pd.read_csv('./winequality-white.csv', sep=';').as_matrix()

def EM(X, k, theta, objective, likelihood, m_step, max_iter=100, print_freq=10):
	r = np.ones((X.shape[0], k)) / k
	pi = np.ones(k) / k
	objectives = [objective(X, r, pi, theta)]

	for i in range(max_iter):
		if (i % print_freq) == 0:
			print("[i={}] objective={}".format(i, objectives[-1]))

		# approximate r
		r = likelihood(X, theta) * pi
		r = r / r.sum(axis=1)[:, np.newaxis]
		# maximize
		pi, theta = m_step(X, r)

		objectives.append(objective(X, r, pi, theta))
	return (objectives, r, pi, theta)

def gmm(X, k, prior_alpha, max_iter=100, print_freq=10):
	S_0 = np.diag(np.std(X, axis=0)**2) / k**(1/X.shape[1])

	Theta = namedtuple("GMM", "mean cov")
	theta = Theta(mean=X[np.random.randint(0, X.shape[0], (k,))], cov=np.tile(S_0, (k, 1, 1)),)

	def likelihood(X, theta):
		p = np.zeros((X.shape[0], k))
		for i in range(k):
			p[:, i] = scipy.stats.multivariate_normal.pdf(X, theta.mean[i], theta.cov[i] + 1e-4*np.eye(X.shape[1]),)
		return p

	denom = X.shape[0] + prior_alpha.sum() - k
	prior_nu = X.shape[1] + 2

	def m_step(X, r):
		r_sum = r.sum(axis=0)
		pi = (r_sum + prior_alpha - 1) / denom
		mu = ((X[:,:,np.newaxis] * r[:, np.newaxis, :]).sum(axis=0) / r_sum).transpose()
		sigma = np.zeros((k, X.shape[1], X.shape[1]))
		for i in range(k):
			diff = (X - mu[i]) * np.sqrt(r[:, i])[:, np.newaxis]
			diff = np.matrix(diff)
			sigma[i] = (diff.transpose() * diff + S_0) / (prior_nu + r_sum[i] + X.shape[1] + 2)
		return pi, Theta(mean=mu, cov=sigma)

	def objective(X, r, pi, theta):
		log_prior = sum(np.log(scipy.stats.invwishart.pdf(theta.cov[i], df=prior_nu, scale=S_0,)) \
			for i in range(k)) + np.log(scipy.stats.dirichlet.pdf(pi, alpha=prior_alpha))
		pi_term = (r * np.log(pi)[np.newaxis, :]).sum()
		likelihood_term = r * np.log(likelihood(X, theta))
		likelihood_term = likelihood_term[r > 1e-12].sum()
		return likelihood_term + pi_term + log_prior

	return EM(X, k, theta, objective, likelihood, m_step, max_iter=max_iter, print_freq=print_freq)

X = np.concatenate((red, white), axis=0)
y = np.zeros((X.shape[0],))
y[:red.shape[0]] = 1

k = 2
obj, r, pi, theta = gmm(X, k, np.ones(2), max_iter=30, print_freq=10)
y_pred = np.argmax(r, axis=1)
# print y_pred.item(5)

def plotGen(obj, r, pi, theta, k=2):
	# plt.style.use('bmh')
	plt.figure(1)
	plt.title('2 Component MAP GMM')
	plt.xlabel('# Iteration')
	plt.ylabel('Log likelihood')
	plt.plot(obj, 'r')
	# plt.savefig('convergence_3.pdf', format='pdf')

	plt.figure(2)
	plt.rcParams.update(plt.rcParamsDefault)
	plt.style.use('default')
	# y_pred = np.argmax(r, axis=1)
	plt.imshow(metrics.confusion_matrix(y, ~y_pred.astype(bool),), cmap=plt.cm.gray_r)
	plt.title('GMM Confusion Matrix')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.colorbar()

	ticks = np.arange(2)
	classes = ['white', 'red']
	plt.xticks(ticks, classes)
	plt.yticks(ticks, classes)
	plt.tight_layout()
	plt.savefig('confusion_3.pdf', format='pdf')
	plt.show()

plotGen(obj, r, pi, theta)

