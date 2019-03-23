# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:38:07 2018

@author: gela
"""

import numpy as np
import theano as theako
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from Nafo.basic_functions import error_rate, y2ind



'''


THIS IS GOING TO BE VERY PAINFUN 

PAINFUN...


'''



class Convoutional_Neural_net(object):

    
    def get_shape(self):
        shape = self.shape_x
        for i in range(len(self.CNN)):
            filter_w_h = np.int32(self.CNN[i][0][2])
            shape = shape - filter_w_h + 1
            shape = np.floor(shape/2) #if pool size changes it should be noted here
            
            shape_squared = shape**2
             
           # shape = np.int32(shape*layer[0][0])
        self.shape_ann_first = np.int32(shape_squared*self.CNN[i][0][0])
        return None
                
    def __init__(self, CNN = [], ANN = [], shape_x = 128):
        #WORKS AS EXPECTED
        
    
        '''
        
        FOR CNN ONLY
        
        SO I EXPECT THERE TO BE LIST OF TUPLES EACH ONE CONTAINING
        
        (*shape, activation, ws)
        
        EXAMPLE
        
        ((10, 1, 5, 5), T.nnet.relu, (2,2))
        
        IT MEANS 10 filters with one color channel, width = 5, height = 5
        
        '''
        self.weights_optimized = False
        self.CNN = CNN
        self.ANN = ANN
        self.shape_x = shape_x
        self.get_shape()
        '''
        
        initialize CNN filters first with numpy random arrays
        
        '''
        self.ws_init_cnn = []
        
        #number_of_conv_layers = len(self.CNN)
        
        for layer in self.CNN:
            shape = layer[0] # (N, C, W, H)
            N = shape[0]
            C = shape[1]
            W = shape[2]
            H = shape[3]
            
            # N is number of filters
            # C is color channels 
            # W = width
            # H = height
            
            temp_initial_weights = np.abs(np.random.randn(N,C,W,H)/(100))
            temp_initial_bias = np.abs(np.zeros(N, dtype = np.float32))
            self.ws_init_cnn.append((temp_initial_weights.astype(np.float32), temp_initial_bias))
            

       #now initialize theano variables, shared
        
        
        self.ws_theano_cnn = []
        self.dws_theano_cnn = []
        i = 0
        for weight in self.ws_init_cnn:
            theano_shared_weight = theako.shared(weight[0], 'filter ' + str(i))
            theano_shared_bias = theako.shared(weight[1], 'bias ' + str(i))
                                               
            self.ws_theano_cnn.append((theano_shared_weight, theano_shared_bias))
            shape_w = weight[0].shape
            shape_b = weight[1].shape
            self.dws_theano_cnn.append((theako.shared(np.zeros(shape_w).astype(np.float32)),
                                        theako.shared(np.zeros(shape_b).astype(np.float32))))
            
            i += 1
            
        '''
        
        THEANO SHARED VARS ARE NOW INITIALIZED FOR CNN
        
        '''
        
        self.ws_init_ann = []
        i = 0
        for layer in self.ANN:
            shape = layer[0]
            N, D = shape
            if i == 0:

                
                
                temp_init_weight_ann = (np.random.randn(self.shape_ann_first,D)).astype(np.float32)
                temp_init_bias_ann = np.zeros(D, dtype = np.float32)
                self.ws_init_ann.append((temp_init_weight_ann, temp_init_bias_ann))
            
            else:
                temp_init_weight_ann = np.random.randn(N,D).astype(np.float32)
                temp_init_bias_ann = np.zeros(D).astype(np.float32)
                self.ws_init_ann.append((temp_init_weight_ann, temp_init_bias_ann))
     
            i += 1
            
        #end loop    
            
    
        self.ws_theano_ann = [] 
        self.dws_theano_ann = []
        
        for weight in self.ws_init_ann:
            theano_shared_weight = theako.shared(weight[0])
            theano_shared_bias = theako.shared(weight[1])
            self.ws_theano_ann.append((theano_shared_weight, theano_shared_bias))
            shape_w = weight[0].shape
            shape_b = weight[1].shape
            self.dws_theano_ann.append((theako.shared(np.zeros(shape_w).astype(np.float32)),
                                        theako.shared(np.zeros(shape_b).astype(np.float32))))
            
    
        del self.ws_init_ann
        del self.ws_init_cnn
        del i 
        del layer
        del weight
        
        #self.ws_theano_cnn
        #  [(filter 0, bias 0), (filter 1, bias 1)] ...
        #self.ws_theano_ann
        #  [(<TensorType(float32, matrix)>, <TensorType(float32, vector)>)] ...
        
        #also self.dw_theano cnn and ann
        return None
        
        
    
    def fit(self, 
            X, 
            Y,
            lr_cnn = 1e-6, 
            lr_ann = 1e-6, 
            mu_cnn = 0.99,
            mu_ann = 0.99,
            epoch = 200,
            batch_size = 100,
            print_period = 100):
        
        lr_cnn = np.float32(lr_cnn)
        lr_ann = np.float32(lr_ann)
        mu_ann = np.float32(mu_ann)
        mu_cnn = np.float32(mu_cnn)
        nbatches = int(np.floor(len(X)/batch_size))

        
        
        '''
        
        
        now I should create theano map for convoutional neural net
        
        it must be more PAINFUN I think...
        
        
        '''
        Y2 = y2ind(Y, len(set(Y))).astype(np.int32)
         
        #THIS SI to BE IMPLEMENTED
        
        '''
        
        conv = T.ReLU(X.conv(W1) + b.dimshuffle())
        pooled = maxpool(conv, (2,2))
        
        repeat same
        
        
        flatten
        
        fully connected neural network
        
        '''
        
        len_cnn = len(self.CNN)
        
        #self.ws_theano_cnn
        #  [(filter 0, bias 0), (filter 1, bias 1)] ...
        #self.ws_theano_ann
        #  [(<TensorType(float32, matrix)>, <TensorType(float32, vector)>)] ...
        
        #also self.dw_theano cnn and ann
        
        Xth = T.tensor4('gela',  dtype = 'float32')
        Yth = theako.tensor.matrix('Y', dtype = 'int32')

        conv_out = T.nnet.conv2d(Xth,self.ws_theano_cnn[0][0])
        pooled = pool_2d(conv_out, ws =self.CNN[0][2], ignore_border = True, mode = 'max')
        activated = self.CNN[0][1](pooled + self.ws_theano_cnn[0][1].dimshuffle('x', 0, 'x', 'x'))
        
        for i in range(1, len_cnn - 1):
                conv_out = T.nnet.conv2d(activated, self.ws_theano_cnn[i][0])
                pooled = pool_2d(conv_out, ws = self.CNN[i][2], ignore_border = True, mode = 'max')
                activated = self.CNN[i][1](pooled  + self.ws_theano_cnn[i][1].dimshuffle('x', 0, 'x', 'x'))


        conv_out = T.nnet.conv2d(activated, self.ws_theano_cnn[len_cnn - 1][0])
        pooled = pool_2d(conv_out, ws = self.CNN[len_cnn - 1][2], ignore_border = True, mode = 'max')
        activated = self.CNN[len_cnn - 1][1](pooled + self.ws_theano_cnn[len_cnn - 1][1].dimshuffle('x', 0, 'x', 'x'))      
        
        '''
        
        NOW ANN PART
        
        
        '''
            
        Z = self.ANN[0][1](activated.flatten(ndim = 2).dot(self.ws_theano_ann[0][0]) + self.ws_theano_ann[0][1])   
            
       
        for i in range(1, len(self.ANN)):
           dot_prod = Z.dot(self.ws_theano_ann[i][0]) + self.ws_theano_ann[i][1]
           Z = self.ANN[i][1](dot_prod) #activation
         
        #last layer should be fostmax
        
       # Z = T.log(Z)
        
        cost = -(Yth*T.log(Z)).sum()
        
        '''
        now weight updates
        '''
        
        updates_array = []
        
        for i in range(0, len(self.CNN)):
                
                gradient_weight_cost = T.grad(cost, self.ws_theano_cnn[i][0])
                gradient_bias_cost = T.grad(cost, self.ws_theano_cnn[i][1])
            
                W_up = self.ws_theano_cnn[i][0] + mu_cnn*self.dws_theano_cnn[i][0] - lr_cnn*gradient_weight_cost
                b_up = self.ws_theano_cnn[i][1] + mu_cnn*self.dws_theano_cnn[i][1] - lr_cnn*gradient_bias_cost
                dW_up = W_up - self.ws_theano_cnn[i][0]
                db_up = b_up - self.ws_theano_cnn[i][1]
                updates_array.append((self.ws_theano_cnn[i][0], W_up))
                updates_array.append((self.ws_theano_cnn[i][1], b_up))
                updates_array.append((self.dws_theano_cnn[i][0], dW_up))
                updates_array.append((self.dws_theano_cnn[i][1], db_up))

       ##    graded = W_up.eval()
       # print(graded)

        for i in range(0, len(self.ANN)):
            
                gradient_weight_cost = T.grad(cost, self.ws_theano_ann[i][0])
                gradient_bias_cost = T.grad(cost, self.ws_theano_ann[i][1])            
            
                W_upa = self.ws_theano_ann[i][0] + mu_ann*self.dws_theano_ann[i][0] - lr_ann*gradient_weight_cost
                b_upa = self.ws_theano_ann[i][1] + mu_ann*self.dws_theano_ann[i][1] - lr_ann*gradient_bias_cost
                dW_upa = W_upa - self.ws_theano_ann[i][0]
                db_upa = b_upa -self.ws_theano_ann[i][1]
                
                updates_array.append((self.ws_theano_ann[i][0], W_upa))
                updates_array.append((self.ws_theano_ann[i][1], b_upa))
                updates_array.append((self.dws_theano_ann[i][0], dW_upa))
                updates_array.append((self.dws_theano_ann[i][1], db_upa))
        
    
        Y_hat = Z.argmax(axis = 1)
        
        predict = theako.function(inputs = [Xth, Yth], outputs = [Y_hat, cost], on_unused_input = 'ignore')
        train = theako.function(inputs = [Xth, Yth], updates = updates_array, on_unused_input = 'ignore')
      #   get_filter = theako.function(inputs = [], outputs = self.ws_theano_cnn[0])
     
      #  FILT = self.ws_theano_ann[1][1].eval()
      #  print(FILT)
        
        for i in range(0, epoch):
            train(X, Y2)
            for j in range(nbatches):
                X_batch = X[j*nbatches:j*nbatches + nbatches]
                Y_batch = Y2[j*nbatches:j*nbatches + nbatches]
                train(X_batch, Y_batch)
                if i % print_period == 0:
                    print('iteration ' + str(i) + ' batch ' + str(j + 1))
                    print('error rate ')
                    pred, cst = predict(X, Y2)
                    err = error_rate(Y, pred)
                    print(err)
                    print(cst)


        self.weights_optimized = True
        return pred
       




    def predict(self, X, Y = 0):
        #WORKS AS EXPECTED 
        if self.weights_optimized == False:
            print('Weights are not optimised !!!')
            return 0
                 
        Xth_pred = T.tensor4('gela1',  dtype = 'float32')

        len_cnn = len(self.CNN)
        conv_out = T.nnet.conv2d(Xth_pred,self.ws_theano_cnn[0][0])
        pooled = pool_2d(conv_out, ws =self.CNN[0][2], ignore_border = True, mode = 'max')
        activated = self.CNN[0][1](pooled + self.ws_theano_cnn[0][1].dimshuffle('x', 0, 'x', 'x'))
        
        for i in range(1, len_cnn - 1):
                conv_out = T.nnet.conv2d(activated, self.ws_theano_cnn[i][0])
                pooled = pool_2d(conv_out, ws = self.CNN[i][2], ignore_border = True, mode = 'max')
                activated = self.CNN[i][1](pooled  + self.ws_theano_cnn[i][1].dimshuffle('x', 0, 'x', 'x'))


        conv_out = T.nnet.conv2d(activated, self.ws_theano_cnn[len_cnn - 1][0])
        pooled = pool_2d(conv_out, ws = self.CNN[len_cnn - 1][2], ignore_border = True, mode = 'max')
        activated = self.CNN[len_cnn - 1][1](pooled + self.ws_theano_cnn[len_cnn - 1][1].dimshuffle('x', 0, 'x', 'x'))      
        
        '''
        
        NOW ANN PART
        
        
        '''
            
        Z = self.ANN[0][1](activated.flatten(ndim = 2).dot(self.ws_theano_ann[0][0]) + self.ws_theano_ann[0][1])   
            
       
        for i in range(1, len(self.ANN)):
           dot_prod = Z.dot(self.ws_theano_ann[i][0]) + self.ws_theano_ann[i][1]
           Z = self.ANN[i][1](dot_prod) #activation
         
        #last layer should be fostmax

        pred = Z.argmax(axis = 1)
        
        predict_op = theako.function(inputs = [Xth_pred], outputs = [pred])
        
        prediction = predict_op(X)
        if type(Y) != type(0):
            error = error_rate(prediction, Y)
            print('error rate')
            print(error)

        return prediction
    
    def set_weights(self, CNN_weights_bias, ANN_weights_bias):
        #NOT TESTED YET
        
        '''        
                
        AQ VELIT 
        
        CNN = [ (filter, bias ), (filter, bias) ... ]
        
        ANN = [ (weight, bias), (weight, bias) ... ]
        
        
        '''
        
        self.CNN_weights_bias = CNN_weights_bias
        self.ANN_weights_bias = ANN_weights_bias
        
        #self.ws_theano_cnn
        #self.ws_theano_ann
        
        if len(self.CNN_weights_bias) != len(self.ws_theano_cnn):
            print('Numbers of layers do not match in Convoutional Leyer!!!')
            return 0

        if len(self.ANN_weights_bias) != len(self.ws_theano_ann):
            print('Numbers of layers do not match in Convoutional Leyer!!!')
            return 0            
        
        
        for i in range(len(self.ws_theano_cnn)):
            filt = self.CNN_weight_bias[i][0]
            bias = self.CNN_weight_bias[i][1]
            
            self.ws_theano_cnn[i][0].set_value(filt)
            self.ws_theano_cnn[i][1].set_value(bias)
            
        '''
        
        NOW FOR ANN LAYER
        
        '''
            
        for i in range(len(self.ws_theano_ann)):
            weight = self.ws_theano_ann[i][0]
            bias = self.ws_theano_ann[i][1]
            
            self.ws_theano_ann[i][0].set_value(weight)
            self.ws_theano_ann[i][1].set_value(bias)            
            
        self.weights_optimized = True
        print('Sucesfully set values of filters weights and biases', end = '\n')
        self.weights_optimized = True
        return 0
    
    
    
    def get_weights(self, makefile = True):
        #WORKS AS EXPECTED
        
        '''
        even though you could just grab weights since they are not private variabled
        it is nice practice to be able to get these weights with a funcion so I will not be confused
        
        jsut returns self.cnn and self.ann
        

   
        AQ VABRUNEBT
        
        CNN = [ (filter, bias ), (filter, bias) ... ]
        
        ANN = [ (weight, bias), (weight, bias) ... ]
        
        
        
        NOTE THAT IT RETURNS FLATTENED COEFFICIENTS, IT CAN ALSO CREATE DAT 
        
        '''
        cnn_weights_array = []
        ann_weights_array = []
        
        for i in range(len(self.ws_theano_cnn)):
            filt = self.ws_theano_cnn[i][0]
            bias_filt = self.ws_theano_cnn[i][1]
            
            filt = filt.flatten(ndim = 2)
            
            
            filt_np = np.array(filt.eval()).astype(np.float32)
            bias_filt_np = np.array(bias_filt.eval()).astype(np.float32)
            
            cnn_weights_array.append((filt_np, bias_filt_np))
            
            if makefile:
                filt_np.tofile('filt' + str(i) + '.dat')
                bias_filt_np.tofile('bias_filt' + str(i) + '.dat')
                


            
        for j in range(len(self.ws_theano_ann)):
            weight = self.ws_theano_ann[j][0]
            bias = self.ws_theano_ann[j][1]
            
            weight_np = np.array(weight.eval()).astype(np.float32)
            bias_np = np.array(bias.eval()).astype(np.float32)
            
            ann_weights_array.append((weight_np, bias_np))   
            
            if makefile:
                
                weight_np.tofile('weight' + str(j) + '.dat')
                bias_np.tofile('bias_weight' + str(j) + '.dat')
                    
        print('SUCESFULLY RETREIVED COEFFICIENTS')
        return cnn_weights_array, ann_weights_array


    def load_from_files(self, datatype = np.float32):
        
        '''
        this function is a bit harder since  you 
        (I) also have to take shapes into consideration...
        '''
        
        
        for i in range(len(self.CNN)):
            shape = self.CNN[i][0] #returns shape of filters 
            N = shape[0]
            C = shape[1]
            W = shape[2]
            H = shape[3]
            
            filter_np = np.fromfile('filt' + str(i) + '.dat', dtype = datatype) 
            filter_np = filter_np.reshape(N, C, W, H)
            bias_filt_np = np.fromfile('bias_filt' + str(i) + '.dat', dtype = datatype)
            
            self.ws_theano_cnn[i][0].set_value(filter_np)
            self.ws_theano_cnn[i][1].set_value(bias_filt_np)
            
            
            
            
        for j in range(0, len(self.ANN)):
            
            if j == 0:
                shape = (self.shape_ann_first, self.ANN[j][0][1]) #shape is ('x', N) self.ann_shape_first to be added
                m, n = shape
            else:
                
                shape = self.ANN[j][0]
                m, n = shape

            
            weight_np = np.fromfile('weight' + str(j) + '.dat', dtype = datatype)
            weight_np = weight_np.reshape(m, n)
            bias_weight_np = np.fromfile('bias_weight' + str(j) + '.dat', dtype = datatype)
            
            self.ws_theano_ann[j][0].set_value(weight_np)
            self.ws_theano_ann[j][1].set_value(bias_weight_np)
            
        self.weights_optimized = True

        print('Weights loaded sucesfully')
        print("You can run predict function or grab weights written in arrays")



'''


CONGRATULATIONS...

get weights works and cretes files

load weights works 

fit function works (gradient descent) 

predict function works 

init also works 

shapes are set correctly 


ravi meti araferia wesit iseti ho ?

'''
    
    
    
    
    
    
