import numpy as np
import matplotlib.pyplot as plt

mu, sigma, sampleSize = 0, 1, 100
s = np.random.normal(mu, sigma, sampleSize)

m_s = 62.0 / 35
b_s = 18.0 / 35

s_x = np.linspace(0, 10, num=sampleSize)

s_y = []

for i in range(sampleSize):
	s_y.append(m_s * s_x[i] + b_s + s[i])

X = np.matrix([(1, x) for x in s_x])
Y = np.matrix([(y) for y in s_y])

ans = np.linalg.inv(X.transpose() * X) * X.transpose() * Y.transpose()
b_s2, m_s2 = ans.item(0), ans.item(1)

l_x = range(100)
l_y = []
for i in range(len(l_x)):
	l_y.append(m_s2 * l_x[i] + b_s2)

plt.plot(s_x, s_y, 'ro')
plt.plot(l_x, l_y)

for i in range(len(l_x)):
	l_y[i] = m_s * l_x[i] + b_s

plt.plot(l_x, l_y)
plt.axis([0, 10, 0, 20])
plt.show()


print ans


