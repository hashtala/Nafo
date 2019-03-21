# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:09:47 2018

@author: gela
"""
import numpy as np



def shuffle(X,Y, shuffles = 100):

    X = np.append(X, Y.reshape(-1,1).astype(np.float32),axis = 1)
    for i in range(shuffles):
                
        index1 = np.random.randint(0, len(X))
        index2 = np.random.randint(0, len(X))
        temp_array = X[index1]
        X[index1] = X[index2]
        X[index2] = temp_array

    
    X_shuffed = X[:,:-1]
    Y_shuffled = X[:,-1]
    return X_shuffed.astype(np.float32), Y_shuffled.astype(np.int32)

def y2ind(Y, K):
    N = len(Y)
    Z = np.zeros((N, K), dtype = np.int32)
    for gela in range(N):
        Z[gela, Y[gela]] = 1
        
    return Z

def sigmoid_plus(X):
    return 1/ (np.exp(-20*X))


def tanh_plus(X):
    return np.tan(20*X)

def tanh(X):
    return np.tanh(X)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))



def relu_plus(X):
    return 20*X*(X > 0)


def relu(X):
    return X*(X > 0)

def softmax(X):
    expX = np.exp(X)
    softX = expX/expX.sum(axis = 1, keepdims = True)
    return softX


def initialize_weigh_bias(m,n):
    W = (np.random.rand(m,n)/np.sqrt(m+n)).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    
    return W, b

def initialize_filter(shape, pool_sz):
    W = np.random.rand(*shape)/(np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:])/np.prod(pool_sz)))
    return W.astype(np.float32)

def sigmoid_cost(T, Y):
    cost = -(T*np.log(Y) + (1 - T)*np.log(1 - Y)).sum()
    #this is binary cross entropy 
    return cost

def cost(T,Y):
    return -(T*np.log(Y)).sum()

def cost2(T,Y):
    #instead of indicator matrices we use vectors to compute cost
    cost = -np.log(Y[np.arange(len(Y)), T]).sum()
    return cost


def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def classification_rate(targets, predictions):
    return 1 - error_rate(targets, predictions)

def error_rate_ish_linear(Y, Yhat):
    dif =  np.abs(Y - Yhat)
    return dif.sum()





     
    


