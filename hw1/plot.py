import numpy as np
import matplotlib.pyplot as plt

mu, sigma, sampleSize = 0, 1, 100
s = np.random.normal(mu, sigma, sampleSize)

m_s = 62.0 / 35
b_s = 18.0 / 35

plt.plot(s_x, s_y, 'ro')
plt.plot(l_x, l_y)
plt.axis([0, 100, 0, 200])
plt.show()

print ans


