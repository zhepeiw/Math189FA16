import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import copy

img = Image.open("/Users/FernandoWang/Downloads/hw2pr5.jpg")

imgMat = np.array(list(img.getdata(band=0)), float)
imgMat.shape = (img.size[1], img.size[0])
imgMat = np.matrix(imgMat)
# plt.imshow(imgMat, cmap='gray')
# plt.show()

U, sigma, V = np.linalg.svd(imgMat)
print sigma


def plotGen():
	plt.figure(1)
	plt.style.use('ggplot')
	# plt.subplot(211)
	orderedPlot, = plt.plot(range(100), sigma[:100], 'bD')
	plt.title('singular values')

	shuffledSigma = copy.deepcopy(sigma)
	random.shuffle(shuffledSigma)
	# shuffledSigma = np.matrix(shuffledSigma)
	# print shuffledSigma
	# reconstimg = np.matrix(U[:,:499] * np.diag(shuffledSigma[:499]) * np.matrix(V[:499,:]))
	# plt.subplot(211)
	shuffledPlot, = plt.plot(range(100), shuffledSigma[:100], 'r^')
	plt.legend((orderedPlot, shuffledPlot), ("100 largest","shuffled"), loc=1)
	# plt.imshow(reconstimg, cmap='gray')



	plt.figure(2)
	plt.rcParams.update(plt.rcParamsDefault)
	# plt.style.use('ggplot')
	plt.subplot(221)
	plt.imshow(imgMat, cmap='gray')
	plt.title('Original')

	plt.subplot(222)
	reconstimg = np.matrix(U[:,:2] * np.diag(sigma[:2]) * np.matrix(V[:2, :]))
	plt.imshow(reconstimg, cmap='gray')
	plt.title('k = 2')

	plt.subplot(223)
	reconstimg = np.matrix(U[:,:10] * np.diag(sigma[:10]) * np.matrix(V[:10, :]))
	plt.imshow(reconstimg, cmap='gray')
	plt.title('k = 10')

	plt.subplot(224)
	reconstimg = np.matrix(U[:,:20] * np.diag(sigma[:20]) * np.matrix(V[:20, :]))
	plt.imshow(reconstimg, cmap='gray')
	plt.title('k = 20')

	plt.show()

plotGen()
