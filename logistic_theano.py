# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 20:46:06 2019

@author: gela
"""

import theano as theako
import theano.tensor as T
import numpy as np
from Nafo.basic_functions import error_rate
import matplotlib.pyplot as plt



class Logistic_Regression():
    def __init__(self, activation):
        self.act = activation #we expect T.nnet.sigmoid 
        
        
    def fit(self, X, Y, learning_rate = 0.00001, epoch = 10e4, print_period = 500):
        N, M = X.shape
        W_initial = np.random.rand(M)
        b_initial = np.random.rand(N)
        
        
        Xth = theako.tensor.matrix('X', dtype = 'float64')
        Yth = theako.tensor.vector('Y', dtype = 'int32')
        
        
        W_thean = theako.shared(W_initial, 'weight')
        b_thean = theako.shared(b_initial, 'bias')
        
        
        Y_pred = T.nnet.sigmoid(Xth.dot(W_thean) + b_thean).round()
        cost = -(Yth*np.log(Y_pred)).sum()
        
        
        W_updt = W_thean - learning_rate*T.grad(cost, W_thean)
        b_updt = b_thean - learning_rate*T.grad(cost, b_thean)
                
        
        train = theako.function(inputs = [Xth, Yth], 
                                updates = [(W_thean, W_updt),
                                           (b_thean, b_updt)
                                ],
                                on_unused_input='ignore')
       
        error_rate_arry = []
        for step in range(int(epoch)):
            train(X, Y)
            if step % print_period == 0:
                print('iteration', str(step))
                erorr = error_rate(Y, Y_pred)
                error_rate_arry.append(erorr)
                print(str(erorr) + '- error rate')
                

                
        return Y_pred, error_rate_arry
        
        
    def predict(self, X):
        return self.act(X.dot(self.w) + self.b)
        




r1 = 5
r2 = 10

x = np.linspace(0, 2*np.pi, 50)
sin = np.sin(x)
cos = np.cos(x)

Circle_1_x = r1*cos
Circle_1_y = r1*sin
Circle1 = np.array([Circle_1_x,Circle_1_y]).T
plt.scatter(Circle1[:, 0], Circle1[:,1])



Circle_2_x = r2*cos
Circle_2_y = r2*sin
Circle2 = np.array([Circle_2_x,Circle_2_y]).T
plt.scatter(Circle2[:, 0], Circle2[:,1])

plt.scatter(Circle1[:,0], Circle1[:, 1], c = 'yellow')
plt.scatter(Circle2[:,0], Circle2[:, 1], c = 'green')

X = np.append(Circle1, Circle2, axis = 0)
Y = np.array([0]*50 + [1]*50)
        
anoth_dim = (X[:,0]**2 + X[:,1]**2).reshape(-1,1) #this is r^2
X = np.append(X, anoth_dim, axis = 1)

Neuron = Logistic_Regression(T.nnet.sigmoid)
pred, cost = Neuron.fit(X, Y) 
