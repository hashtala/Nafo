# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:54:40 2019

@author: gela
"""

import tensorflow as tf
import numpy as np


'''


UNDA DAEMATOS ROM X SHAPE AGAR GADAEWOOS DISCRIMINATOR VARIABLE INITIALIZERS

DA PIRDAPIR X-is DIMENSION GADAEWODOS RA


'''




class DCGAN(object):
    
    
    def __init__(self,
                 X,
                 lr_disc = 10e-6,
                 lr_gen = 10e-3,
                 batch = 100,
                 activation = tf.nn.sigmoid,
                 projection = 128,
                 startdim = 7,
                 isvecyot = False,
                 latent_dim = 200
                 ):
        
        self.X_shape = X.shape
        self.batch = batch
        self.activation = activation
        self.latent_dim = latent_dim
        self.projection = projection
        self.startdim = startdim
        self.isvecyot = isvecyot
        self.lr_disc = lr_disc
        self.lr_gen = lr_gen
        self.Generator_variables_creator()
        self.discriminator_variables_initializer(X)



        
        self.Xtheano = tf.placeholder(dtype = np.float32)
        generated_images = self.graph_and_get_image(self.batch, True)
        generated_images_logits = self.graph_and_logits(generated_images)
        real_logits = self.graph_and_logits(self.Xtheano)
        
        d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits = real_logits,
                                                              labels = tf.ones_like(real_logits))
        
        d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(logits = generated_images_logits,
                                                              labels = tf.zeros_like(generated_images_logits)) 
        

        self.d_cost = tf.reduce_mean(d_cost_real) + tf.reduce_mean(d_cost_fake)
       
        g_cost_ish = tf.nn.sigmoid_cross_entropy_with_logits(logits = generated_images_logits,
                                                            labels = tf.ones_like(generated_images_logits))
       
        
        self.g_cost = tf.reduce_mean(g_cost_ish)
        
        
        real_pred = tf.cast(real_logits > 0, tf.float32)
        fake_pred = tf.cast(generated_images_logits < 0, tf.float32)
        num_preds = 2.0*self.batch
        correct_preds = tf.reduce_sum(real_pred) + tf.reduce_sum(fake_pred)
        self.accuracy = correct_preds/num_preds
        
        
        self.discriminator_parametes = [param for param in tf.trainable_variables() if param.name.startswith('D')]
        self.generator_parameters = [p for p in tf.trainable_variables() if p.name.startswith('G')]
        
        self.d_train_operation = tf.train.AdamOptimizer(self.lr_disc, 0.5, 0.99).minimize(self.d_cost, var_list = self.discriminator_parametes)
        self.g_train_operation = tf.train.AdamOptimizer(self.lr_gen, 0.5, 0.99).minimize(self.g_cost, var_list = self.generator_parameters)
    
      
        
        self.init_op = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        
    def discriminator_variables_initializer(self, X, isvecyot = False):
    
        x_shape = X.shape
      
        
        with tf.variable_scope('Discriminator_conv', reuse = tf.AUTO_REUSE):
            
            #fitler height, filter width, input maps, output maps
            init_filter1 = np.random.randn(3, 3, 1, 10).astype(np.float32)
            self.filter1 = tf.get_variable('filte1_disc', initializer = init_filter1)
            init_bias1 = np.zeros(10).astype(np.float32)
            self.bias1 = tf.get_variable('bias1_disc', initializer = init_bias1)
            
            
            init_filter2 = np.random.randn(3, 3, 10, 10).astype(np.float32)
            self.filter2 = tf.get_variable('filte2_disc', initializer = init_filter2)
            init_bias2 = np.zeros(10).astype(np.float32)
            self.bias2 = tf.get_variable('bias2_disc', initializer = init_bias2)
            
            
            init_filter3 = np.random.randn(3, 3, 10, 10).astype(np.float32)
            self.filter3 = tf.get_variable('filte3_disc', initializer = init_filter3)
            init_bias3 = np.zeros(10).astype(np.float32)
            self.bias3 = tf.get_variable('bias3_disc', initializer = init_bias3)
            
            
        first_ann_shape = int(np.ceil(x_shape[1]/2))
        first_ann_shape = int(np.ceil(first_ann_shape/2))
        first_ann_shape = int(np.ceil(first_ann_shape/2))
        
        
        
        #IF IT IS A VECTOR IT SHOULD BE NOTED HERE
        if not isvecyot:
            first_ann_shape = first_ann_shape*first_ann_shape*10 
        
            
        
        
        
        with tf.variable_scope('Discriminnator_ann', reuse = tf.AUTO_REUSE):
            
            init_weight1_ann = np.random.randn(first_ann_shape, 800).astype(np.float32)
            self.weight1_ann = tf.get_variable('ann_first', initializer = init_weight1_ann)
            init_bias1_ann = np.zeros(800).astype(np.float32)
            self.bias1_ann = tf.get_variable('baias1', initializer = init_bias1_ann)
            
            
            init_weight2_ann = np.random.randn(800 , 800).astype(np.float32)
            self.weight2_ann = tf.get_variable('ann_second', initializer = init_weight2_ann)
            init_bias2_ann = np.zeros(800).astype(np.float32)
            self.bias2_ann = tf.get_variable('baias2', initializer = init_bias2_ann)
            
            
            return None
            
            
    def graph_and_logits(self, X):
      
        
        #X_tf = tf.placeholder(dtype = np.float32)   
    
            
        
        first_conv = tf.nn.conv2d(X, self.filter1, strides = [1, 2, 2, 1], padding = 'SAME')
        first_conv = tf.nn.bias_add(first_conv, self.bias1)
        first_conv = tf.nn.relu(first_conv)
        
        
        
        second_conv = tf.nn.conv2d(first_conv, self.filter2, strides = [1, 2, 2, 1], padding = 'SAME')
        second_conv = tf.nn.bias_add(second_conv, self.bias2)
        second_conv = tf.nn.leaky_relu(second_conv)
        
        third_conv = tf.nn.conv2d(second_conv, self.filter3,  strides = [1, 2, 2, 1], padding = 'SAME')
        third_conv = tf.nn.bias_add(third_conv, self.bias3)
        third_conv = tf.nn.relu(third_conv)
        
        
        flattened = tf.contrib.layers.flatten(third_conv)
        
        first_ann = tf.matmul(flattened, self.weight1_ann) + self.bias1_ann
        first_ann = tf.nn.relu(first_ann)
        logits = tf.matmul(first_ann, self.weight2_ann) + self.bias2_ann

        
      #  init_op = tf.initialize_all_variables()
       # sess = tf.Session()
      #  sess.run(init_op)
#        j = sess.run(logits)
        
        return logits



    def Generator_variables_creator(self):
    
        
      

    
        last_dim_neural = self.startdim*self.startdim*self.projection
        
        
        with tf.variable_scope('Generator_ANN', reuse = tf.AUTO_REUSE):
            init_weight_gen_ann = np.random.randn(self.latent_dim,last_dim_neural).astype(np.float32)
            self.weight1_gen_ann = tf.get_variable('jela', initializer = init_weight_gen_ann)
            init_bias_gen_ann = np.zeros(last_dim_neural, dtype = np.float32)
            self.bias_gen_ann = tf.get_variable('jela_bias', initializer = init_bias_gen_ann)
     
            
            
           
    
        #now let's create filters 
        
        with tf.variable_scope('Generator_conv', reuse = tf.AUTO_REUSE):
            init_filt1_gen = np.random.randn(5,5, int(self.projection/2), self.projection).astype(np.float32) #outputs N, 8, 8, 70
            self.filt1_gen = tf.get_variable('filt1_gen', initializer = init_filt1_gen)
            init_bias1_gen = np.random.randn(int(self.projection/2)).astype(np.float32)
            self.bias1_gen = tf.get_variable('bias1_gen', initializer = init_bias1_gen)
            
            init_filt_bridge = np.random.randn(5,5,int(self.projection/4),int(self.projection/2)).astype(np.float32)
            self.filter_bridge = tf.get_variable('filter_bridged_gen', initializer = init_filt_bridge)
            init_bias2_bridged = np.zeros(int(self.projection/4)).astype(np.float32)
            self.bias_bridged = tf.get_variable('bias_bridged', initializer =init_bias2_bridged)
            
            init_filt_bridge2 = np.random.randn(5,5,int(self.projection/8),int(self.projection/4)).astype(np.float32)
            self.filter_bridge2 = tf.get_variable('filter_bridged_gen2', initializer = init_filt_bridge2)
            init_bias2_bridged2 = np.zeros(int(self.projection/8)).astype(np.float32)
            self.bias_bridged2 = tf.get_variable('bias_bridged2', initializer =init_bias2_bridged2)            
            
            
            
            del init_filt1_gen, init_bias1_gen, init_filt_bridge, init_bias2_bridged
            
            init_filt2_gen = np.random.randn(5,5, 1,int(self.projection/8)).astype(np.float32) #outputs N, 8, 8, 70
            self.filt2_gen = tf.get_variable('filt2_gen', initializer = init_filt2_gen)
            init_bias2_gen = np.random.randn(1).astype(np.float32)
            self.bias2_gen = tf.get_variable('bias2_gen', initializer = init_bias2_gen)
            
            del init_filt2_gen, init_bias2_gen
        

        
        
        
        return None
            
    def graph_and_get_image(self, n, is_training = True):
        
        Z = np.random.uniform(-1, 1, size = (n, self.latent_dim)).astype(np.float32)
        
        z_dot_w1 = tf.nn.relu(tf.matmul(Z, self.weight1_gen_ann) + self.bias_gen_ann)
        gela = tf.nn.relu(tf.matmul(z_dot_w1, self.weigh2_gen_ann) + self.bias2_gen_ann)
        
        projected = tf.reshape(gela, [-1, self.startdim, self.startdim, self.projection])
        #print(projected.shape)
        projected = tf.contrib.layers.batch_norm(projected, is_training = is_training)
        projected = tf.nn.relu(projected)
        
    
        
        first_convolved = tf.nn.conv2d_transpose(projected, self.filt1_gen, output_shape = (self.batch,14,14,int(self.projection/2)), strides = [1, 2, 2,1], padding = 'SAME')
        first_convolved = tf.nn.bias_add(first_convolved, self.bias1_gen)
        first_convolved = tf.contrib.layers.batch_norm(first_convolved, is_training = is_training)
        first_convolved = tf.nn.relu(first_convolved)
        #print(first_convolved.shape)
        
        
        brige_convovle = tf.nn.conv2d_transpose(first_convolved, self.filter_bridge, output_shape = (self.batch, 14, 14, int(self.projection/4)), strides =[ 1, 1,1 ,1], padding = 'SAME')
        brige_convovle = tf.nn.bias_add(brige_convovle, self.bias_bridged)
        brige_convovle = tf.contrib.layers.batch_norm(brige_convovle, is_training = is_training)
        brige_convovle = tf.nn.relu(brige_convovle)
        #print(brige_convovle.shape)
        
        brige_convovle2 = tf.nn.conv2d_transpose(brige_convovle, self.filter_bridge2, output_shape = (self.batch, 14, 14, np.int32(self.projection/8)), strides =[ 1, 1,1 ,1], padding = 'SAME')
        brige_convovle2 = tf.nn.bias_add(brige_convovle2, self.bias_bridged2)
        brige_convovle2 = tf.contrib.layers.batch_norm(brige_convovle2, is_training = is_training)
        brige_convovle2 = tf.nn.relu(brige_convovle2)
        #print(brige_convovle2.shape)
        
        #print(self.filt1_gen.shape)
        #print(self.filter_bridge.shape) 
        #print(self.filter_bridge2.shape) 
       # print(self.filt2_gen.shape)
        
        almost_image = tf.nn.conv2d_transpose(brige_convovle2, self.filt2_gen, output_shape = (self.batch, 28, 28, 1), strides = [1, 2, 2, 1], padding = 'SAME')
        almost_image = tf.nn.bias_add(almost_image, self.bias2_gen)
        #print(almost_image.shape)
        
 

        #almost_image = tf.contrib.layers.batch_norm(almost_image, is_training = is_training)
        #init = tf.global_variables_initializer()
        ##sess = tf.Session()
        #sess.run(init)
        almost_image = self.activation(almost_image)
    
        
        #init_op = tf.initialize_all_variables()
        #sess = tf.Session()
        #sess.run(init_op)
        '''
        out1 = sess.run(z_dot_w1)
        out2 = sess.run(gela)
        proj = sess.run(projected)
        convolved = sess.run(first_convolved)
        '''
        #image = sess.run(almost_image)
        
        return almost_image




    def fit(self, X, epoch = 2):


        #self.define_costs_params()  
        
        
        
        
        n_batches = self.X_shape[0]//self.batch
        
        for i in range(epoch):
            np.random.shuffle(X)
            for j in range(n_batches):
                X_batch = X[j*self.batch: j*self.batch + self.batch]
                
                _, d_cost, accuracy = self.sess.run((self.d_train_operation, self.d_cost, self.accuracy), feed_dict = {self.Xtheano: X_batch})
                
                _, g_cost = self.sess.run((self.g_train_operation, self.g_cost))
                _, g_cost = self.sess.run((self.g_train_operation, self.g_cost))
                _, g_cost = self.sess.run((self.g_train_operation, self.g_cost))

         
                print('iteration ' + str(i) + ' batch ' + str(j))
                print('Accuracy: ' + str(accuracy))
                print('D cost ' +str(d_cost))
                print('G cost ' +str(g_cost))


               
               
        return self.graph_and_get_image(self.batch, False)   
                

    def load_weight(self):
        discriminator_weights = self.sess.run(self.discriminator_parametes)
        generator_weights = self.sess.run(self.generator_parameters)
        
        discriminator_weights[0].tofile('filt1_disc1.dat')
        discriminator_weights[1].tofile('bias1.dat')
        discriminator_weights[2].tofile('filt1_disc2.dat')
        discriminator_weights[3].tofile('bias2.dat')        
        discriminator_weights[4].tofile('filt1_disc3.dat')
        discriminator_weights[5].tofile('bias3.dat')
        discriminator_weights[6].tofile('ann_disc1.dat')
        discriminator_weights[7].tofile('ann_bias1.dat')
        discriminator_weights[8].tofile('ann_disc2.dat')
        discriminator_weights[9].tofile('ann_bias2.dat')


        generator_weights[0].tofile('gen_ann1.dat')
        generator_weights[1].tofile('gen_bias_ann1.dat')
        generator_weights[2].tofile('gen_ann2.dat')
        generator_weights[3].tofile('gen_bias_ann2.dat')
        generator_weights[4].tofile('transpose_filter1.dat')
        generator_weights[5].tofile('gen_bias_transpose1.dat')
        generator_weights[6].tofile('transpose_filter2.dat')
        generator_weights[7].tofile('gen_bias_transpose.dat')










X = np.random.randn(201, 28, 28, 1).astype(np.float32)
S = DCGAN(X)
j = S.graph_and_logits(X)
im = S.graph_and_get_image(100)
nana = S.fit(X)

S.discriminator_variables_initializer(X)

#preds = graph_and_get_image(100, False)
