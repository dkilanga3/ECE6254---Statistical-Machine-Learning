import numpy as np
import matplotlib.pyplot as plt



mu = 0
sigma = 1
n = 1000
Z = []
for index in range(n):
	x = np.random.normal(mu, sigma,1000000)
	Z.append(max(x))


plt.hist(Z)
plt.title("Histogram of Z with Beta = 6")

## Show the plot
plt.show()
