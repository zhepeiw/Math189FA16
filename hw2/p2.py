import math
import numpy as np
import matplotlib.pyplot as plt

def configPlot(ax):
	ax.spines['left'].set_position('center')
	ax.spines['bottom'].set_position('center')

	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')

	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

x1 = np.linspace(-1.0, 0, num=50)
y1 = np.linspace(0, 1.0, num=50)
y2 = np.linspace(0, -1.0, num=50)

x2 = np.linspace(0, 1.0, num=50)
y3 = np.linspace(1.0, 0, num=50)
y4 = np.linspace(-1.0, 0, num=50)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
configPlot(ax)

# p = 2
circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='#cccccc')
ax.add_patch(circ)

# p = 1
plt.style.use('ggplot')
plt.plot(x1, y1, zorder=99)
plt.plot(x1, y2, zorder=99)
plt.plot(x2, y3, zorder=99)
plt.plot(x2, y4, zorder=99)

ax.fill_between(x1, y2, y1, facecolor='None', hatch="X", interpolate=True, zorder=99)
ax.fill_between(x2, y4, y3, facecolor='None', hatch="X", interpolate=True, zorder=99)

plt.title("p = 1: square / p = 2: circle")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()