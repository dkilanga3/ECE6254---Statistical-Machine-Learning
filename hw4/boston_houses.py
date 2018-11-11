import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
boston = load_boston()
X = boston.data
y = boston.target



X_train = X[0:400,:]
X_test = X[400:, :]
y_train = y[0:400]
y_test = y[400:]


X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

X_train_mean = np.reshape(X_train_mean, (1,13))
X_train_std = np.reshape(X_train_std, (1,13))

for index in range(X_train.shape[0]):
	X_train[index,:] = np.divide(np.subtract(X_train[index,:], X_train_mean), X_train_std)
	if index < 106:
		X_test[index,:] = np.divide(np.subtract(X_test[index,:], X_train_mean), X_train_std)


#X_train_mean = np.mean(X_train, axis=0)
#X_train_std = np.std(X_train, axis=0)
#X_train_var = np.var(X_train, axis=0)
#print X_train_mean
#print X_train_var
#print X_train_std


##########################################################################################################
# Question 1.a

A = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)), y_train)

#A_T_A_inv = np.linalg.inv(A_T_A)
#A_T_A_inv_A_T = np.dot(A_T_A_inv, np.transpose(A))
#theta = np.dot(A_T_A_inv_A_T, y_train)
X_test_tilde = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
MSE_LS = 1/float(X_test.shape[0])*np.dot(np.transpose(y_test - np.dot(X_test_tilde, theta)), (y_test - np.dot(X_test_tilde, theta)))


#test = y_test - np.dot(X_test_tilde, theta)
#test1 = np.dot(np.transpose(test), test)
print 'MSE Least Square: ', MSE_LS

############################################################################################################
# Question 1.b

X_train_fit = X_train[0:300,:]
X_train_hold = X_train[300:,:]
y_train_fit = y_train[0:300]
y_train_hold = y_train[300:]

#Lambda = 0.001

A1 = np.concatenate((np.ones((X_train_fit.shape[0], 1)), X_train_fit), axis=1)
A1_T_A1 = np.dot(np.transpose(A1),A1);
#while Lambda < 2:
#	theta1 = np.dot(np.dot(np.linalg.inv(A1_T_A1 + Lambda * np.identity(A1_T_A1.shape[0])), np.transpose(A1)), y_train_fit)
#	X_train_hold_tilde = np.concatenate((np.ones((X_train_hold.shape[0], 1)), X_train_hold), axis=1)

# Lambda = 1.5 is the value that minimize this MSE, so we will use it on our
#	MSE_R_hold = 1/float(X_train_hold.shape[0])*np.dot(np.transpose(y_train_hold - np.dot(X_train_hold_tilde, theta1)), (y_train_hold - np.dot(X_train_hold_tilde, theta1)))
#	print MSE_R_hold, Lambda
#	Lambda += 0.001


# Getting theta using all the training data and lambda = 1.507
Lambda = 1.507
A_T_A = np.dot(np.transpose(A), A)
theta1 = np.dot(np.dot(np.linalg.inv(A_T_A + Lambda * np.identity(A_T_A.shape[0])), np.transpose(A)), y_train)

# Compute MSE for ridge regression

MSE_R = 1/float(X_test.shape[0])*np.dot(np.transpose(y_test - np.dot(X_test_tilde, theta1)), (y_test - np.dot(X_test_tilde, theta1)))

print 'MSE Ridge Regression: ', MSE_R

################################################################################################################
# Question 1.c

# uncomment this part to run the code used to deterine Lambda
#Lambda =0.001
#from sklearn import linear_model
#while Lambda < 10:
#	reg = linear_model.Lasso(alpha = Lambda)
#	reg.fit(np.concatenate((np.ones((X_train_fit.shape[0], 1)),X_train_fit), axis=1), y_train_fit)
#	MSE_Lasso = 1/float(X_train_fit.shape[0])*np.dot(np.transpose(y_train_hold - reg.predict(X_train_hold_tilde)), (y_train_hold - reg.predict(X_train_hold_tilde)))
#	print MSE_Lasso, Lambda
#	Lambda += 0.001
# The best Lambda minimizing the MSE on the hold data I was able to compute is 3.966

#Lambda = 3.966
Lambda = 0.17
from sklearn import linear_model
reg = linear_model.Lasso(alpha = Lambda)
reg.fit(np.concatenate((np.ones((X_train.shape[0], 1)),X_train), axis=1), y_train)
MSE_Lasso = 1/float(X_test.shape[0])*np.dot(np.transpose(y_test - reg.predict(X_test_tilde)), (y_test - reg.predict(X_test_tilde)))
number_none_zero_theta = np.count_nonzero(reg.coef_)

print 'MSE Lasso: ' , MSE_Lasso
print 'number of non zero theta', number_none_zero_theta
