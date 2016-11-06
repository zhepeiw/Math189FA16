import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from collections import namedtuple

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

def bernoullis(X, k, prior_alpha, prior_a, prior_b, max_iter=50, print_freq=10):
	S_0 = np.diag(np.std(X, axis=0)**2) / k**(1/X.shape[1])
	Theta = namedtuple('BMM', 'mean')

	theta = Theta(mean=X[:k*np.floor(X.shape[0] / k)].reshape(k, -1, X.shape[1],).mean(axis=1),)

	def likelihood(X, theta):
		p = np.tile(theta.mean.T, (X.shape[0], 1, 1))
		p[X == 0] = 1 - p[X == 0]
		p = p.prod(axis=1)
		return p

	denom = X.shape[0] + prior_alpha.sum() - k
	def m_step(X, r):
		r_sum = r.sum(axis=0)
		pi = (r_sum + prior_alpha - 1) / denom
		mu = (((X[:,:,np.newaxis] * r[:, np.newaxis, :]).sum(axis=0) + prior_a - 1) / \
			(r_sum + prior_a + prior_b - 2))
		mu = mu.transpose()
		return pi, Theta(mean=mu)

	def objective(X, r, pi, theta):
		log_prior = np.log(scipy.stats.beta.pdf(theta.mean, prior_a, prior_b,)).sum() + \
		np.log(scipy.stats.dirichlet.pdf(pi, alpha=prior_alpha), )
		pi_term = (r * np.log(pi)[np.newaxis, :]).sum()
		likelihood_term = r * np.log(likelihood(X, theta))
		likelihood_term = likelihood_term[r > 1e-12].sum()
		return likelihood_term + pi_term + log_prior

	return EM(X, k, theta, objective, likelihood, m_step, max_iter=max_iter, print_freq=print_freq,)

train = pd.read_csv('./mnist_train.csv', header=None)
X = train.iloc[:, 1:].as_matrix()
X = (X > X[X > 0].mean()).astype(float)
y = train.iloc[:, 0].as_matrix()
del train

np.random.seed(1)
N = int(10000)
subset_ix = np.random.randint(0, X.shape[0], (N,))
smallX = X[subset_ix]

k = 10
obj, r, pi, theta = bernoullis(smallX, k, prior_alpha=np.ones(10), \
	prior_a=1, prior_b=1, max_iter=50, print_freq=10)

def genConvergencePlot(obj):
	
	plt.style.use('bmh')
	plt.plot(obj, 'r')
	plt.title('MAP bernoullis Mixture')
	plt.xlabel('# iteration')
	plt.ylabel('log likelohood')
	plt.tight_layout()
	plt.savefig('convergence_4_b.pdf', format='pdf')
	plt.show()

def genMeanPlot(theta):
	plt.figure(figsize=(5,2))
	for i in range(10):
		plt.subplot(2, 5, i + 1)
		img = theta.mean[i]
		plt.imshow(img.reshape(28, 28), cmap='Greys')
		plt.axis('off')

	plt.suptitle('Means of Mixture of Bernoullis')
	plt.tight_layout()
	plt.savefig('mean_4.pdf', format='pdf')
	# plt.show()

# genMeanPlot(theta)
genConvergencePlot(obj)

