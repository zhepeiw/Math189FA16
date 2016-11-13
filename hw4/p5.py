import numpy as np
import copy
import matplotlib.pyplot as plt

def NMF(V, r, max_iter=200):
	n = V.shape[0]
	m = V.shape[1]
	obj = []
	# W: n x r, H: r x m
	W, H = np.mean(V) * np.matrix(np.random.rand(n, r)), np.mean(V) * np.matrix(np.random.rand(r, m))
	for iter in range(max_iter):
		Wcopy = copy.deepcopy(W)
		Hcopy = copy.deepcopy(H)
		# update H
		for row in range(r):
			for col in range(m):
				# s = (W.transpose() * V)
				# print row, col, (W.transpose() * W * H).shape, n, r
				Hcopy[row, col] = H.item(row, col) * ((W.transpose() * V).item(row, col)) \
				/ ((W.transpose() * W * H).item(row, col))
		# update W
		for row in range(n):
			for col in range(r):
				Wcopy[row, col] = W.item(row, col) * ((V * H.transpose()).item(row, col)) \
				/ ((W * H * H.transpose()).item(row, col))
		W, H = Wcopy, Hcopy
		obj.append(dist(V, W * H))

	
	return W, H, [obj[i] for i in range(len(obj)) if i % 2 == 0]

def dist(V, U):
	assert(V.shape == U.shape)
	diff = 0
	for i in range(V.shape[0]):
		for j in range(V.shape[1]):
			diff += (V.item(i, j) - U.item(i, j))**2
	return np.sqrt(diff)

test = [[1.1,1.2,1.3,1.4], [1.8,1.7,0.6,0.5],[1.5,1.6,2.7,0.8], [2.3,2.2,1.1,1.4], [0.9,1.4,3.2,2.4]]
V = np.matrix(test)
W, H, obj = NMF(V, 3)
print W

def genCvgPlot(obj):
	plt.style.use('bmh')
	plt.figure(1)
	plt.title('NMF Convergence Plot')
	plt.xlabel('# Iteration')
	plt.ylabel('RMSE')
	plt.plot(obj, 'r')
	plt.savefig('confusion_5.pdf', format='pdf')
	plt.show()
# genCvgPlot(obj)
