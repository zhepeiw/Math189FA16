import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open("/Users/FernandoWang/Downloads/hw2pr5.jpg")

imgMat = np.array(list(img.getdata(band=0)), float)
imgMat.shape = (img.size[1], img.size[0])
imgMat = np.matrix(imgMat)
plt.imshow(imgMat, cmap='gray')
plt.show()

U, sigma, V = np.linalg.svd(imgMat)

kArr = [2, 10, 20]

reconstimg = np.matrix(U[:,:1] * np.diag(sigma[:1]) * np.matrix(V[:1, :]))

def plotGen():
	plt.figure(1)
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
