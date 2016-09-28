# =======problem 2=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, sparse, misc
from sklearn.preprocessing import OneHotEncoder
# from scipy import sparse
print "shiiit"
df = pd.read_csv('/Users/FernandoWang/Downloads/mnist_train.csv', sep=',', engine='python').as_matrix()
df_test = pd.read_csv('/Users/FernandoWang/Downloads/mnist_test.csv', sep=',', engine='python').as_matrix()
# df.describe()
# ========part a========
dataBin = [row for row in df if row[0] == 0 or row[0] == 1]
dataBin = np.matrix(dataBin)
x_train_bin = dataBin[:,1:]
y_train_bin = dataBin[:, 0]

testBin = [row for row in df_test if row[0] == 0 or row[0] == 1]
testBin = np.matrix(testBin)
X_test_bin = testBin[:,1:]
y_test_bin = testBin[:, 0]

def linreg(X, y, reg=0.0):
	eye = np.eye(X.shape[1])
	eye[0,0] = 0. # don't regularize bias term
	X = np.matrix(X)
	y = np.matrix(y)
	return np.linalg.solve(X.transpose() * X + reg * eye, X.transpose() * y)

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def log_likelihood(X, y_bool, theta, reg=1e-6):
	X = np.matrix(X)
	# theta = np.matrix(theta)
	mu = sigmoid(X * theta / X.shape[0] / 10)
	mu[~y_bool] = 1 - mu[~y_bool]
	# ans = np.log(mu).sum() - reg*np.inner(theta, theta)/2
	# print 'ans', reg*np.inner(theta, theta)
	# return (np.log(mu).sum() - reg*np.inner(theta, theta)/2).item(0)
	return (np.log(mu).sum() - reg*np.inner(theta, theta)/2).item(0)

def log_likelihood_newton(X, y_bool, theta, reg=1e-6):
	X = np.matrix(X)
	# theta = np.matrix(theta)
	mu = sigmoid(X * theta)
	mu[~y_bool] = 1 - mu[~y_bool]
	# ans = np.log(mu).sum() - reg*np.inner(theta, theta)/2
	# print 'ans', reg*np.inner(theta, theta)
	# return (np.log(mu).sum() - reg*np.inner(theta, theta)/2).item(0)
	return (np.log(mu).sum() - reg*np.inner(theta, theta)/2).item(0)

def grad_log_likelihood(X, y, theta, reg=1e-6):
	X = np.matrix(X)
	theta = np.matrix(theta)
	return X.transpose() * (sigmoid(X * theta) - y) + reg * theta

# def newton_step(X, y, theta, reg=1e-6):
# 	X = np.matrix(X)
# 	mu = np.array(m.item(0) for m in sigmoid(X * theta))
# 	# mu = sigmoid(X * theta)
# 	# mu = np.matrix(mu)
# 	diagElement = (mu.transpose() * (1 - mu)).item(0)
# 	# print mu.transpose() * (1 - mu)

# 	# using a cholesky solve is exactly twice as fast as a regular np.linalg.solve.
# 	# also, using scipy.sparse.diags will be much more efficient than constructing
# 	# the entire diagonal scaling matrix. Same with sparse.eye.
# 	# fack1 = X.transpose() * sparse.diags([diagElement], offsets=0, shape=X.shape)
# 	fack1 = (diagElement * np.eye(X.shape[1])) * X.transpose() 
# 	# print fack1
# 	fack2 = fack1 * X
# 	fack3 = reg * sparse.eye(X.shape[1])
# 	# print fack1.shape, fack2.shape, fack3.shape
# 	fack4 = fack2 + fack3
# 	return linalg.cho_solve(linalg.cho_factor(fack2 + fack3),grad_log_likelihood(X, y, theta, reg=reg))

def newton_step(X, y, theta, reg=1e-6):
	X = np.matrix(X)
	mu = np.array([m.item(0) for m in sigmoid(X * theta)])
	return linalg.cho_solve(		
		linalg.cho_factor(X.transpose() * sparse.diags(mu * (1 - mu), 0) * X + reg * sparse.eye(X.shape[1])),
		grad_log_likelihood(X, y, theta, reg=reg),
	)

def lr_grad(X, y, reg=1e-6, lr=1e-3, tol=1e-6, max_iters=300, verbose=False, print_freq=5,):
	y = y.astype(bool)
	y_bool = np.array([yy.item(0) for yy in y], dtype=bool)

	# y_bool = [x.item(0) for x in y]
	# y_bool = np.array(y, dtype = bool)
	# y_bool = np.array([yy.item(0) for yy in y], dtype=bool)


	theta = np.matrix(np.zeros(X.shape[1])).transpose()
	objective = [log_likelihood(X, y_bool, theta, reg=reg)]
	grad = grad_log_likelihood(X, y, theta, reg=reg)
	while len(objective)-1 <= max_iters and np.linalg.norm(grad) > tol:
		if verbose and (len(objective)-1) % print_freq == 0:
			print('[i={}] likelihood: {}. grad norm: {}'.format(len(objective)-1, objective[-1], np.linalg.norm(grad),))
		grad = grad_log_likelihood(X, y, theta, reg=reg)
		theta = theta - lr * grad
		objective.append(log_likelihood(X, y_bool, theta, reg=reg))
	if verbose:
		print('[i={}] done. grad norm = {:0.2f}'.format(len(objective)-1, np.linalg.norm(grad)))
	return theta, objective

def lr_newton(X, y,reg=1e-6, tol=1e-6, max_iters=300,verbose=False, print_freq=1,):
	y = y.astype(bool)
	y_bool = [x.item(0) for x in y]
	y_bool = np.array(y_bool, dtype = bool)

	theta = np.matrix(np.zeros(X.shape[1])).transpose()
	objective = [log_likelihood_newton(X, y_bool, theta, reg=reg)]
	step = newton_step(X, y, theta, reg=reg)

	while len(objective)-1 <= max_iters and np.linalg.norm(step) > tol:
		if verbose and (len(objective)-1) % print_freq == 0:
			print('[i={}] likelihood: {}. step norm: {}'.format(len(objective)-1, objective[-1], np.linalg.norm(step)))
		step = newton_step(X, y, theta, reg=reg)
		theta = theta - step
		objective.append(log_likelihood_newton(X, y_bool, theta, reg=reg))
	if verbose:
		print('[i={}] done. step norm = {:0.2f}'.format(len(objective)-1, np.linalg.norm(step)))
	return theta, objective

def get_accuracy_for_lambda(l):
	theta_newton, objective_newton = lr_newton(x_train_bin, y_train_bin, 
		max_iters=300, reg=l)
	y_predicted = X_test_bin * theta_newton

	for i in range(len(y_predicted)):
		if y_predicted[i] > 0.:
			y_predicted[i] = 1
		else:
			y_predicted[i] = 0

	accu = accuracy(y_test_bin, y_predicted)
	print 'Newton accuracy: {}'.format(accu)
	return accu

def accuracy(y_test, y_predict):
	correct = 0
	for i in range(len(y_test)):
		if (y_test[i] == y_predict[i]):
			correct += 1
	return correct * 1. / len(y_test)

def plotGenA():
	thetaGrad, objGrad = lr_grad(x_train_bin, y_train_bin,max_iters=500, verbose=True)
	thetaNewton, objNewton = lr_newton(x_train_bin, y_train_bin, max_iters=500, verbose=True)
	# print 'grad', len(objGrad)
	# print 'newton', len(objNewton)

	# print [np.linalg.norm(o) for o in objGrad]
	# print [np.linalg.norm(o) for o in thetaGrad]
	plt.style.use('ggplot')
	gradPlot, = plt.plot([x for x in range(len(objGrad))], objGrad, 'b')
	newtonPlot, = plt.plot([x for x in range(len(objNewton))], objNewton, 'r')
	plt.title('likelihood vs iteration')
	plt.xlabel('iteration')
	plt.ylabel('likelihood')
	plt.legend((gradPlot, newtonPlot), ('Gradient Descent', 'Newton'), loc=4)
	plt.show()


def plotGenAccu():
	ls = [1e-6] + [3.0 * i for i in range(1, 10)]
	plt.style.use('ggplot')
	acctPlot = plt.plot(ls, [get_accuracy_for_lambda(l) for l in ls], color='blue')	
	plt.ylim([0, 1.1])	
	plt.title('Accuracy vs Lambda')
	plt.xlabel('Accuracy')
	plt.ylabel('Lambda')

	plt.show()

# ========part b============
def softmax(x):
	s = np.exp(x - np.max(x, axis = 1))
	return s / np.sum(s, axis=1)

def log_softmax(x):
	return x - misc.logsumexp(x, axis=1)

def softmax_log_likelihood(X, y_one_hot, W, reg=1e-6):
	X = np.matrix(X)
	W = np.matrix(W)
	W_Transpose = np.transpose(W)

	mu = X * W
	return np.sum(mu[y_one_hot] - misc.logsumexp(mu, axis =1)) - \
	reg * np.einsum('ij,ji->', W_Transpose, W)/2

def soft_grad_log_likelihood(X, y_one_hot, W, reg=1e-6):
	X = np.matrix(X)
	print X.shape
	X_Transpose = np.transpose(X)
	W = np.matrix(W)
	print W.shape
	mu = X * W
	mu = np.exp(mu- np.max(mu, axis=1))
	mu = mu / np.sum(mu, axis=1)
	return X_Transpose * (mu-y_one_hot) + reg*W

def softmax_grad(
	X, y, reg=1e-6, lr=1e-8, tol=1e-6,
	max_iters=300, batch_size=256,
	verbose=False, print_freq=5):

	enc = OneHotEncoder()
	y_one_hot = enc.fit_transform(y.copy().reshape(-1,1)).astype(bool).toarray()
	W = np.zeros((X.shape[1], y_one_hot.shape[1]))
	ind = np.random.randint(0, X.shape[0], size=batch_size)
	objective = [softmax_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg)]
	grad = soft_grad_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg)

	while len(objective)-1 <= max_iters and np.linalg.norm(grad) > tol:
		if verbose and (len(objective)-1) % print_freq == 0:
			print('[i={}] likelihood: {}. grad norm: {}'.format(len(objective)-1, \
				objective[-1], np.linalg.norm(grad)))

		ind = np.random.randint(0, X.shape[0], size=batch_size)
		grad = soft_grad_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg)
		W = W - lr * grad

		objective.append(softmax_log_likelihood(X[ind], y_one_hot[ind], W, reg=reg))

	print('[i={}] done. grad norm = {:0.2f}'.format(
		len(objective)-1, np.linalg.norm(grad)
		))

	return W, objective

def genPlotB():
	w, objective = softmax_grad(x_train_bin, y_train_bin, max_iters=500)

	plt.style.use('ggplot')
	plt.ylim([-8500, 500])
	plt.xlim([-20,520])
	bPlot, = plt.plot([x for x in range(len(objective))], objective, color='blue')
	plt.title('Iteration vs Likelihood')
	plt.xlabel('Iteration')
	plt.ylabel('Likelihood')
	# plt.legend((bPlot), ('Likelihood'), loc=3)
	plt.show()

# =========part c============
def predict_knn(X_test, X_train, y_train, k=5):
	num_data = X_test.shape[0]
	y_pred = [0] * num_data

	for i in range(num_data):
		digit = X_test[i]
		index = np.argpartition(1. / np.linalg.norm(X_train - digit[:,np.newaxis].T, axis=1), -k)[-k:]
		y_pred[i] = np.argmax(np.bincount(y_train[index]))

	return y_pred

def generateReport():
	num = 2500
	X_train_sample = df[:num,1:]
	Y_train_sample = df[:num,0]
	X_test_sample = df_test[:num,1:]
	Y_test_sample = df_test[:num,0]

	for k in [1,5,10]:
		print('[k={}] accuracy: {}'.format(
			k,
			accuracy(Y_test_sample, predict_knn(X_test_sample, X_train_sample, Y_train_sample, k=k)),
		))

# generateReport()