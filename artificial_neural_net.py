# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:32:15 2018

@author: gela
"""

import theano.tensor as T
import theano as theako
import numpy as np
from Nafo.basic_functions import y2ind, error_rate, shuffle 


def weight_init_with_bias(m,n):
    W = np.random.rand(m,n)*np.sqrt(2.0/m)
    b = np.random.rand(n)/np.sqrt(n)
    return W.astype(np.float32), b.astype(np.float32)

class HiddenLayer(object):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        W, b = weight_init_with_bias(m,n)
        self.W = theako.shared(W, 'Weight')
        self.b = theako.shared(b, 'bias')
         
        return None
        #I think everythin should be clear here 
    def return_layer(self):
        return (self.W, self.b)
    
class Artificial_Neural_Net(object):
    def __init__(self, 
                 hidden_layer_sizes_funcs):
        #it takes list of tuples, and tules contain m and n of hidden
        #matrix + activation function
        self.hidden = []
        self.activations = []
        self.hidden_momentum = []
        for Hidden in hidden_layer_sizes_funcs:
            m, n = Hidden[0], Hidden[1] #sizes of hidden layers
            pre_hidden = HiddenLayer(m, n) #it initializs object
            hidden_layer = pre_hidden.return_layer() #returns weight and bias
            self.hidden_momentum.append((theako.shared(np.zeros((m,n), dtype = np.float32), 'delta_Weight'), 
                                         theako.shared(np.zeros(n, dtype = np.float32), 'delta_bias'))) #append this stuff as well though 
            self.activations.append(Hidden[2]) #gets function of activation
            self.hidden.append(hidden_layer) #appends hidden layer weight and bias
        return None
        #so we have something like
        #[(W, b1), (W2, b2), (W3, b3)...]
        #[T.nnet.relu, T.nnet.sigmoid...]
        
    def fit(self, X,Y,
                lr = 0.000003,
                mu = 0.99,
                epoch = 10000,
                print_period = 500,
                step = 100,   
                beta = 2,
                batches = 0,
                shuffle_data = True,
                return_weights = False
                ):
            lr = np.float32(lr)
            mu = np.float32(mu)
            beta = np.float32(beta)
            if batches != 0:
                nbatches = np.floor(len(X)/batches)
                nbatches = np.int32(nbatches)

            if shuffle_data:
                 X, Y = shuffle(X,Y) #this is almost necessary for batch gradient descent 
                
            Y2ind = y2ind(Y, len(set(Y)))
            #now we need to define Z and stuff 
            error_rate_array = [] #it accumulates error rate and returns it 
            Xth = theako.tensor.matrix('gela', dtype = 'float32')
            Yth = theako.tensor.matrix('Y', dtype = 'int32')
            
            Z1 = Xth.dot(self.hidden[0][0]) + self.hidden[0][1]
            Z1 = self.activations[0](Z1) #now appy activation function
            self.Zs = [Z1]
            for index in range(1, len(self.hidden) - 1):
                dot_prod = self.Zs[index - 1].dot(self.hidden[index][0]) + self.hidden[index][1]
                #previous activation dotted with another Ws and added bias
                Z = self.activations[index](dot_prod)
                self.Zs.append(Z)

            G = self.Zs[-1] #softmax on last layer
            G = self.activations[len(self.hidden) - 1](G) #does activation on the last (almost)
            #so it will be either softmax of round() or whatever 
            cost = -(Yth*np.log(G)).sum()
            pred = G.argmax(axis = 1)
            
            #now we should write updates
            updates_list = []
            momentum_updates = []
            lr_s = [] #so each layer gets bigger learning rate to solve the vanishing gradient problem
            gela = True
            if beta != 0:
                for lr_updt in range(1, len(self.hidden) + 1):
                   if gela:
                      lr_s.append(lr)
                      gela = False
                   else:
                     lr_updt = np.float32(lr_updt)
                     new_lr = ((beta*lr_updt)*lr)
                     lr_s.append(new_lr)
               
            if beta == 0:
                lr_s = np.ones(len(self.hidden)).astype(np.float32)
            
            
            
            for count in range(len(self.hidden) - 1):
                lr = lr_s[-count]  #this fella tho
                #we are getting (W, b)
                del_w, del_b = self.hidden_momentum[count]
                W, b = self.hidden[count]
                W_updt = (W - lr*T.grad(cost, W))
                dW_updt = W_updt - W
                b_updt = (b - lr*T.grad(cost, b))      
                dB_updt = b_updt - b
                updates_list.append((W_updt, b_updt))
                momentum_updates.append((dW_updt, dB_updt))
           
          
            #ok now I have things to organize
            # I have [(W1, b1), (W2, b2), (W3, b3)]
            # and    [(W1_updt, b1_updt), (W2_updt, b2_updt)]....
            # AND [(del_w1, del_b1), (del_w2, del_b2)]....
            theako_updates =  []
            for index in range(len(self.hidden)-1):
                w_up = updates_list[index][0]
                b_up = updates_list[index][1]
                dW, db = self.hidden_momentum[index]
                del_Ws_up = momentum_updates[index][0]
                del_bs_up = momentum_updates[index][1] #so I get value of del
                
                weight = self.hidden[index][0]
                bias = self.hidden[index][1]
              
                theako_updates.append((dW, del_Ws_up))
                theako_updates.append((db, del_bs_up))
                theako_updates.append((weight, w_up + mu*dW)) 
                theako_updates.append((bias,  b_up + mu*db))
                

                
            train = theako.function(inputs = [Xth, Yth], updates = theako_updates, on_unused_input = 'ignore')
            predict = theako.function(inputs = [Xth], outputs = [pred], on_unused_input = 'ignore')
            
            '''
            now array is initialized that will store cahnges in W, it will have two
            dimensions, one for storing delta W-s and another for storing delta w for all layers
            so '''
            
            delta_w_arry = []
            for count in range(len(self.activations)-1):
                #layer number is same as how many activations are used...
                delta_w_arry.append([])
            '''
            jobs done whatever was written in previous green comments
            '''
                
            for term in range(0, epoch):
                if batches != 0:
                    #check if regular gradient descent is done instead of batch gradient descent 
                    for j in range(nbatches):
                        X_batches = X[j*batches:j*batches + batches]
                        Y2ind_batches = Y2ind[j*batches:j*batches + batches]
                        train(X_batches, Y2ind_batches)
                        
                        if term % print_period == 0:
                            print('iteration ' + str(term) + ' batch ' + str(j))
                            p = predict(X)
                            batch_error= error_rate(p, Y)                        
                            print('Accuracy: ' + str(100*(1-batch_error)) +'%')
                        if term % step == 0:
                            p = predict(X)
                            error = error_rate(p, Y)
                            error_rate_array.append(error)
                            


                        
                else:
                    train(X, Y2ind)
                    if term % print_period == 0:
                        
                        p = predict(X)
                        print('iteration ' + str(term))
                        error = error_rate(p, Y)
                        print('Accuracy: ' + str(100*(1-error)) +'%')
                    if term % step == 0:
                        p = predict(X)
                        error = error_rate(p, Y) #stores error rate temporarily
                        error_rate_array.append(error)
              #  for gela in range(len(delta_w_arry)):
                   # delta_w_arry[gela].append(self.hidden_momentum[gela][0].get_value())
                    
                

            
            W_s = []
            counter = 1
            for hidd in self.hidden:
                
                W, b = hidd
                W_val = W.get_value()
                b_val = b.get_value()
                W_s.append([W_val, b_val, 'Layer ' + str(counter)])
                counter +=1;
            W_s.pop(-1)
            self.W = W_s 
            
            if return_weights:
                return predict(X), error_rate_array, np.array(W_s)
            else:
                return predict(X), error_rate_array
                
            
            
    def predict(self, X):
          Xth = T.matrix('X')
          Z1 = Xth.dot(self.hidden[0][0]) + self.hidden[0][1]
          Z1 = self.activations[0](Z1) #T.nnet.relu(Z)
          self.Zs = [Z1]
          for index in range(1, len(self.hidden)-1):
             dot_prod = self.Zs[index - 1].dot(self.hidden[index][0]) +self.hidden[index][1]
             #previous activation dotted with another Ws and added bias
             Z = self.activations[index](dot_prod)
             self.Zs.append(Z)
        
          
          G = self.Zs[-1] #softmax on last layer
          G = self.activations[len(self.hidden) - 1](G) #does activation on the last (almost)
          pred = G.argmax(axis = 1)
          predict_func = theako.function(inputs = [Xth], outputs = [pred])
          
          return predict_func(X)
      
      #It works 
        

'''
THIS IS A CLASS Of ARTIFICIAL NEURAL NETWORK
IT TAKES DATA AND SIZE OF HIDDEN LAYERS
ALSO YOU MUST FEED IT WITH ACTIVATION FUNCTIONS
'''
