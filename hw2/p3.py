import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

xArr = np.linspace(-5.0, 5.0, num=100)

norm = [stats.norm.pdf(x) for x in xArr]
lap = [stats.laplace.pdf(x) for x in xArr]

plt.figure(1)
plt.style.use('ggplot')

normPlot, = plt.plot(xArr, norm, 'b')
lapPlot, = plt.plot(xArr, lap, 'r')
plt.legend((normPlot, lapPlot), ("Gaussian","laplace"), loc=1)
plt.title("Standard density")
plt.show()
