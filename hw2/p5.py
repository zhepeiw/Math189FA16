import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import copy

img = Image.open("/Users/FernandoWang/Downloads/hw2pr5.jpg")

imgMat = np.array(list(img.getdata(band=0)), float)


imgMat.shape = (img.size[1], img.size[0])
imgMat = np.matrix(imgMat)

shuffledImgMat = copy.deepcopy(imgMat)
shape0 = shuffledImgMat.shape[0]
shape1 = shuffledImgMat.shape[1]
newShuffledList = []
for i in range(shape0 * shape1):
	newShuffledList.append(shuffledImgMat.item(i))

np.random.shuffle(newShuffledList)
shuffledImgMat = np.matrix(newShuffledList)
shuffledImgMat = np.reshape(shuffledImgMat, (shape0,shape1))

# print shuffledImgMat.item(0)

# print shuffledImgMat.item(0)
# plt.imshow(imgMat, cmap='gray')
# plt.show()
# print 'orig', imgMat.item(0), imgMat.item(1), imgMat.item(2)
# print 'shuffled', shuffledImgMat.item(0), shuffledImgMat.item(1), shuffledImgMat.item(2)

U, sigma, V = np.linalg.svd(imgMat)
US, sigmaS, VS = np.linalg.svd(shuffledImgMat)

print 'orig', sigma.item(0), sigma.item(1), sigma.item(2)
print 'shuffled', sigmaS.item(0), sigmaS.item(1), sigmaS.item(2)


def plotGen():
	plt.figure(1)
	plt.style.use('ggplot')
	# plt.subplot(211)
	orderedPlot, = plt.plot(range(100), sigma[:100], 'b')
	plt.title('singular values')

	shuffledSigma = copy.deepcopy(sigma)
	random.shuffle(shuffledSigma)
	# shuffledSigma = np.matrix(shuffledSigma)
	# print shuffledSigma
	# reconstimg = np.matrix(U[:,:499] * np.diag(shuffledSigma[:499]) * np.matrix(V[:499,:]))
	# plt.subplot(211)
	# shuffledPlot, = plt.plot(range(100), shuffledSigma[:100], 'ro')
	shuffledPlot, = plt.plot(range(100), sigmaS[:100], 'ro')
	plt.legend((orderedPlot, shuffledPlot), ("original","random"), loc=1)
	# plt.imshow(reconstimg, cmap='gray')

	# plt.figure(2)
	# plt.rcParams.update(plt.rcParamsDefault)
	# # plt.style.use('ggplot')
	# plt.subplot(221)
	# plt.imshow(imgMat, cmap='gray')
	# plt.title('Original')

	# plt.subplot(222)
	# reconstimg = np.matrix(U[:,:2] * np.diag(sigma[:2]) * np.matrix(V[:2, :]))
	# plt.imshow(reconstimg, cmap='gray')
	# plt.title('k = 2')

	# plt.subplot(223)
	# reconstimg = np.matrix(U[:,:10] * np.diag(sigma[:10]) * np.matrix(V[:10, :]))
	# plt.imshow(reconstimg, cmap='gray')
	# plt.title('k = 10')

	# plt.subplot(224)
	# reconstimg = np.matrix(U[:,:20] * np.diag(sigma[:20]) * np.matrix(V[:20, :]))
	# plt.imshow(reconstimg, cmap='gray')
	# plt.title('k = 20')

	# plt.tight_layout()
	plt.show()

plotGen()
