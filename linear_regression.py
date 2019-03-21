# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 01:18:37 2019

@author: gela
"""

import numpy as np
from Nafo.basic_functions import error_rate_ish_linear
#import matplotlib.pyplot as plt


class Linear_Regression(object):
    def __init__(self):
        return None
    
    def fit(self, X,
            Y,
            learning_rate = 0.000001,
            epoch = 5000,
            momentum = 0,
            print_period = 250):
        #if print period is 0 it never prints 
        #dont forget that X should come with a column of 1's (constant)
        #this ill be equivalent to bias 
        #cofficient will be automatically calcualated by gradiend descent 
        
        cost_array = []
        N, M = X.shape
        W = np.random.randn(M)/100
        
        for step in range(0, epoch):
            Y_pred = X.dot(W)
            
            derivative = X.T.dot(Y_pred - Y)
            W = W - momentum*W - learning_rate*derivative
            
            if step % (print_period)  == 0:
                #this if else statement ensures that if print period is 
                cost = error_rate_ish_linear(Y_pred, Y)
                cost_array.append(cost)
                print('iterateion', str(step))
                print(cost)
        
        self.W = W
        return X.dot(self.W), cost_array
                
    def predict(self, X):
        Weights = self.W
        return X.dot(Weights)




'''
it works just fine 
there is test data below
to remind myself how it exactly 
can be implemented

dont forget to import matplot lib if you want to test 
'''
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
        
    

Model = Linear_Regression()
pred, costs = Model.fit(X1,Y, epoch = 5000, learning_rate = 0.0005)
plt.plot(costs)
'''