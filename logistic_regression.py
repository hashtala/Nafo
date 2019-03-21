# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:03:09 2019

@author: gela
"""
import numpy as np
import theano
import theano.tensor as T
from Nafo import basic_functions as bsf

class Logistic_Regression(object):
    def __init__(self, dim, activation):
        self.activation = activation 
        W_init = np.random.rand(dim,1)
        b_init = np.random.rand(1)
        self.W = theano.shared(W_init, 'Weights')
        self.b = theano.shared(b_init, 'bias')
        
        
        
    def fit(self,
            X,
            Y, 
            lr = 10e-5,
            epoch = 10e3,
            print_period = 100):
    
        
        Xth = theano.tensor.matrix('Xela', dtype = 'float64')
        TargetsTH = theano.tensor.vector('Yela', dtype = 'int32')
        W = self.W
        b = self.b
        Z = Xth.dot(self.W) + self.b
        Pish = self.activation(Z)
        P = Pish.round()


        cost = -(TargetsTH*np.log(P)).sum()

        W_updt = W - lr*T.grad(cost, W)
        b_updt = b - lr*T.grad(cost, b)
        
        train = theano.function(inputs = [Xth, TargetsTH], updates = [(W, W_updt), (b, b_updt)],  on_unused_input = 'ignore')
        predict = theano.function(inputs = [Xth], outputs = [P],  on_unused_input = 'ignore')
        error_rate_array = []
        for step in range(int(epoch)):
            train(X,Y)
            
            if step % print_period == 0:
                p = predict(X)
                print('iteration ' + str(step + 1))
                error = bsf.error_rate(p, Y) #stores error rate temporarily
                error_rate_array.append(error)
                print('Accuracy: ' + str(100*(1-error)) +'%')
        
        return predict(X), error_rate_array
    
    
    
    
    
r1 = 5
r2 = 10

x = np.linspace(0, 2*np.pi, 50)
sin = np.sin(x)
cos = np.cos(x)

Circle_1_x = r1*cos
Circle_1_y = r1*sin
Circle1 = np.array([Circle_1_x, Circle_1_y]).T



Circle_2_x = r2*cos
Circle_2_y = r2*sin
Circle2 = np.array([Circle_2_x,Circle_2_y]).T


X = np.append(Circle1, Circle2, axis = 0)
Y = np.array([0]*50 + [1]*50)
#now atempt very big neural networ
#then add another dimension and try with one neuron


#now solve it with one neuron 

anoth_dim = (X[:,0]**2 + X[:,1]**2).reshape(-1,1)  #this is r^2
X = np.append(X, anoth_dim, axis = 1)


model = Logistic_Regression(3, T.nnet.sigmoid)
pred, error = model.fit(X,Y, epoch = 10e4, lr = 10e-3, print_period = 1000)