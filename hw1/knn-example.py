import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

np.random.seed(2017) # Set random seed so results are repeatable

n = 100 # number of training points
k = 7 # number of neighbors to consider 

## Generate a simple 2D dataset
X, y = datasets.make_moons(n,'True',0.3)

## Create instance of KNN classifier
classifier = neighbors.KNeighborsClassifier(k,'uniform')
classifier.fit(X, y)

## Plot the decision boundary. 
# Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
h = .02  # step size in the mesh
x_delta = (X[:, 0].max() - X[:, 0].min())*0.05 # add 5% white space to border
y_delta = (X[:, 1].max() - X[:, 1].min())*0.05
x_min, x_max = X[:, 0].min() - x_delta, X[:, 0].max() + x_delta
y_min, y_max = X[:, 1].min() - y_delta, X[:, 1].max() + y_delta
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

## Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-NN classifier trained on %i data points" % (k,n))

## Show the plot
plt.show()
