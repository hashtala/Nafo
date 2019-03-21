# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 19:53:58 2018

@author: gela
"""
import numpy as np
import matplotlib.pyplot as plt
from Nafo import artificial_neural_net as ann
#from linear_regression import Linear_Regression
import theano.tensor as T
from Nafo import basic_functions as bf

X1 = np.random.randn(100,2) + np.array([2,2])
X2 = np.random.randn(100,2) + np.array([-2,-2])
X3 = np.random.randn(100,2) + np.array([4,-4])
X4 = np.random.randn(100,2) + np.array([-4,4])
X = np.vstack([X1,X2, X3, X4])
Y = np.array([0]*100 + [1]*100 + [2]*100 + [3]*100)


Model_Neural = ann.Artificial_Neural_Net([(2, 50, T.nnet.relu),
                                    (50, 50, T.nnet.relu),
                                    (50, 50, T.nnet.relu),
                                    (50, 4, T.nnet.relu),
                                    (1, 1, T.nnet.softmax)]
                                    )


prediction, error_rate = Model_Neural.fit(X, Y, lr = 0.000001, epoch = 3000, print_period = 100)







plt.scatter(X1[:,0], X1[:,1])
plt.scatter(X2[:,0], X2[:,1])
plt.scatter(X3[:,0], X3[:,1])
plt.scatter(X4[:,0], X4[:,1])
plt.show()
plt.plot(error_rate)




'''

X1 = np.array([[1,0],
              [1,1],
              [1,2],
              [1,3],
              [1,4],
              [1,5],
              [1,6],
              [1,7],
              [1,8],
              [1,9],
              [1,10],
              [1,11],
              [1,12],
              [1,13],
              [1,14],
              [1,15],
              
              ]
              )

Y = 2*X1[:,1] + 4
        

Model_Linear = Linear_Regression()
pred, costs = Model_Linear.fit(X1,Y, epoch = 5000, learning_rate = 0.0005)
plt.plot(costs)
'''