import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets 
from math import exp, floor
import time

# the logistic function
def logistic_func(theta, x):
    t = x.dot(theta)
    g = np.zeros(t.shape)
    # split into positive and negative to improve stability
    g[t>=0.0] = 1.0 / (1.0 + np.exp(-t[t>=0.0])) 
    g[t<0.0] = np.exp(t[t<0.0]) / (np.exp(t[t<0.0])+1.0)
    return g

# function to compute log-likelihood
def neg_log_like(theta, x, y):
    g = logistic_func(theta,x)
    return -sum(np.log(g[y>0.5])) - sum(np.log(1-g[y<0.5]))

# function to compute the gradient of the negative log-likelihood
def log_grad(theta, x, y):
    g = logistic_func(theta,x)
    return -x.T.dot(y-g)
    
# implementation of gradient descent for logistic regression
#def grad_desc(theta, x, y, alpha, tol, maxiter):
#    nll_vec = []
#    nll_vec.append(neg_log_like(theta, x, y))
#    nll_delta = 2.0*tol
#    iter = 0
#    while (nll_delta > tol) and (iter < maxiter):
#        theta = theta - (alpha * log_grad(theta, x, y)) 
#        nll_vec.append(neg_log_like(theta, x, y))
#        nll_delta = nll_vec[-2]-nll_vec[-1]
#        iter += 1
#    return theta, np.array(nll_vec), iter

###############################################################################################
###############################################################################################

#Answer 2.b

#def log_hess(theta, x):
#	g = logistic_func(theta,x)
#	hess = 0
#	for index in range(x.shape[0]-1):
#		x_i = x[index,:]
#		hess += x_i.T.dot(x_i.T)*g[index]*(1-g[index])	
#	return hess

#def grad_desc(theta, x, y, tol, maxiter):
#	nll_vec = []
#	nll_vec.append(neg_log_like(theta, x, y))
#	nll_delta = 2.0*tol
#	iter = 0
#	while(nll_delta > tol) and (iter < maxiter):
#		alpha = 1/log_hess(theta, x)
#		theta = theta -(alpha * log_grad(theta, x, y))
#		nll_vec.append(neg_log_like(theta, x, y))
#		nll_delta = nll_vec[-2] - nll_vec[-1]
#		iter += 1
#		#print alpha, nll_vec[-1]
#	return theta, np.array(nll_vec), iter, alpha
###############################################################################################
###############################################################################################	

#Answer 2.c

def grad_desc(theta, x, y, alpha, tol, maxiter):
	nll_vec = []
	data = np.c_[x,y]
	#batch = int(floor(data.shape[0]*0.001))
	batch = 100
	nll_vec.append(neg_log_like(theta, x[0:batch,:], y[0:batch]))
	nll_delta = 2.0*tol
	iter = 0
	while(abs(nll_delta) > tol) and (iter < maxiter):
	#while(condition) and (iter < maxiter):
		#if (iter%30 == 0) and (nll_delta < tol):
			#condition = False
		data = np.random.permutation(data)
		theta = theta - (alpha * log_grad(theta, data[0:batch,:-1], data[0:batch,-1]))
		nll_vec.append(neg_log_like(theta, data[0:batch,:-1], data[0:batch,-1]))
		nll_delta = nll_vec[-2] - nll_vec[-1]
		iter += 1
	return theta, np.array(nll_vec), iter
##################################################################################################
##################################################################################################

# function to compute output of LR classifier
def lr_predict(theta,x):
    # form Xtilde for prediction
    shape = x.shape
    Xtilde = np.zeros((shape[0],shape[1]+1))
    Xtilde[:,0] = np.ones(shape[0])
    Xtilde[:,1:] = x
    return logistic_func(theta,Xtilde)

## Generate dataset    
np.random.seed(2017) # Set random seed so results are repeatable
x,y = datasets.make_blobs(n_samples=100000,n_features=2,centers=2,cluster_std=6.0)

## build classifier
# form Xtilde
shape = x.shape
xtilde = np.zeros((shape[0],shape[1]+1))
xtilde[:,0] = np.ones(shape[0])
xtilde[:,1:] = x

# Initialize theta to zero
theta = np.zeros(shape[1]+1)

# Run gradient descent
alpha = 1
tol = 1e-3
maxiter = 10000
#theta,cost,count = grad_desc(theta,xtilde,y,alpha,tol,maxiter)
#print log_hess(theta, xtilde)
##############################################################################################
##############################################################################################

#Answer to Question 2.a
#while (alpha > 0):
#	theta = np.zeros(shape[1]+1)
#	theta,cost,count = grad_desc(theta,xtilde,y,alpha,tol,maxiter)
#	print 'Alpha: %2.5f  # of interations: %d  Cost: %5.5f' %(alpha, count, cost[-1])
#	alpha -= 0.0001

#print "When alpha is between 1 and 0.0083, there is only one iteration that occurs, and the cost", \
#	" is different for different alphas.\n However, when alpha is less than or equal to 0.0082",\
#	" and greater than zero,\n there are more iterations for each alpha. The alpha resulting in",\
#	" the lowest\n cost is 0.0082"

###############################################################################################
###############################################################################################

#Answer to Question 2.b
#condition = True
#while(condition):
#	theta = np.zeros(shape[1]+1)
#	theta,cost,count, alpha = grad_desc(theta,xtilde,y,tol,maxiter)
#	print 'Alpha: %2.5f  # of interations: %d  Cost: %5.5f' %(alpha, count, cost[-1])
#	if alpha <= 0:
#		condition = False
#print "It takes 37 iterations in odrder to converge"	

################################################################################################
################################################################################################

#Answer to Question 2.c
#alpha = 0.0005
#while(alpha > 0):
#	theta = np.zeros(shape[1]+1)
#	theta,cost,count = grad_desc(theta,xtilde,y,alpha,tol,maxiter)
#	print 'Alpha: %2.5f  # of interations: %d  Cost: %5.5f' %(alpha, count, cost[-1])
#	alpha -= 0.00001
#################################################################################################
#################################################################################################

#Answer to Question 2.d Regular Grad Descent
#alpha = 0.000001
#theta = np.zeros(shape[1]+1)
#time_before = time.time()
#theta,cost,count = grad_desc(theta,xtilde,y,alpha,tol,maxiter)
#time_after = time.time()
#run_time = time_after - time_before
#print 'Run Time: %4.9f  # of interations: %d  Cost: %5.5f' %(run_time, count, cost[-1])

#Answer to Question 2.d Newton's Grad Descent
#theta = np.zeros(shape[1]+1)
#time_before = time.time()
#theta,cost,count, alpha = grad_desc(theta,xtilde,y,tol,maxiter)
#time_after = time.time()
#run_time = time_after - time_before
#print 'Run Time: %5.9f  # of interations: %d  Cost: %5.5f' %(run_time, count, cost[-1])

#Answer to Question 2.d Stochastic Grad Descent
alpha = 0.00001
theta = np.zeros(shape[1]+1)
time_before = time.time()
theta,cost,count = grad_desc(theta,xtilde,y,alpha,tol,maxiter)
time_after = time.time()
run_time = time_after - time_before
print 'Run Time: %5.9f  # of interations: %d  Cost: %5.5f' %(run_time, count, cost[-1])

## Plot the decision boundary. 
# Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
#h = .02  # step size in the mesh
#x_delta = (x[:, 0].max() - x[:, 0].min())*0.05 # add 5% white space to border
#y_delta = (x[:, 1].max() - x[:, 1].min())*0.05
#x_min, x_max = x[:, 0].min() - x_delta, x[:, 0].max() + x_delta
#y_min, y_max = x[:, 1].min() - y_delta, x[:, 1].max() + y_delta
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#Z = lr_predict(theta,np.c_[xx.ravel(), yy.ravel()])

# Create color maps
#cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
#cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Put the result into a color plot
#Z = Z.reshape(xx.shape)
#plt.figure()
#plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

## Plot the training points
#plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)

## Show the plot
#plt.xlim(xx.min(), xx.max())
#plt.ylim(yy.min(), yy.max())
#plt.title("Logistic regression classifier")
#plt.show()
