# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:36:06 2019

@author: gela
"""





import tensorflow as tf
import numpy as np


''' 

I am not mentally read for that yet but...

there we go anyway 

'''



#ok I have figured it out....


class ConvLayer():
    def __init__(self,
                 name,
                 shape_d,
                 batch_norm = False,
                 strides = 2,
                 activation = tf.nn.relu
                 ):
        
        self.name = name
        self.shape = shape_d # shape of filter  (h, w, in, out) 
        self.batch_norm = batch_norm
        self.strides = strides
        self.a = activation
        
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            h, w, imap, omap = self.shape 
            initial_filt = np.random.randn(h, w, imap, omap).astype(np.float32)
            initial_bias = np.zeros(omap).astype(np.float32)
            self.filter = tf.get_variable('filter',
                                          initializer = initial_filt,
                                          trainable = True)
            self.bias = tf.get_variable('bias', 
                                        initializer = initial_bias,
                                        trainable = True)
            
        self.parameter = [self.filter, self.bias]
        
    def Convolution(self,
                    X, 
                    is_training):
        
        conv_out = tf.nn.conv2d(X,
                                self.filter,
                                padding = 'SAME',
                                strides = [1,self.strides,self.strides,1],
                                )
        
        conv_out = tf.nn.bias_add(conv_out, self.bias)
        
        if self.batch_norm:
            conv_out = tf.contrib.layers.batch_norm(inputs = conv_out,
                                                    reuse = tf.AUTO_REUSE,
                                                    scope = self.name,
                                                    scale = True,
                                                    trainable = is_training,
                                                    is_training = is_training
                                                    )
        return self.a(conv_out)
        
        
    
    
    
#img = np.random.randn(1, 28, 28, 3).astype(np.float32)
#shape = (5, 5, 3, 8)
#
#layer = ConvLayer('layer1',
#                  shape = (5, 5, 3, 8),
#                  batch_norm = True
#                  )
#
#out = layer.Convolution(img, True)
#
#
#layer_duplicate = ConvLayer('layer1',
#                            shape = shape)
#
#out2 = layer_duplicate.Convolution(img, True)
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)
#convolved = sess.run(out)
#convolved_duplicate = sess.run(out2)
        
# CONVOLUTION WORKS, IT RE-USES WEIGHTS AND DOES NOT DO FUNNY THINGS : ) 
        
    
class FractionallyStriedConvLayer():
    
    def __init__(self,
                 filter_size,
                 outmap,
                 inmap,
                 name,
                 output_shape,
                 batch_norm = False,
                 strides = 2,
                 activation = tf.nn.relu,
                 ):
        
        self.name = name
        self.output_shape = output_shape
        self.batch_norm = batch_norm
        self.a = activation
        self.strides = strides
        
        '''
        MOKLED AQ ESE XDEBA RA
        
        MOGVDSI OUTPUT SHAPE RA UNDA IYOS
        
        ASEVE X
        
        DA UNDA GAVIGOT FILTRIS SHAPEEBI RA
        
        '''
        
        
        
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            initial_strided_filter = np.random.randn(filter_size, filter_size, outmap, inmap).astype(np.float32)
            self.strided_filter = tf.get_variable('strided_filter',
                                                  initializer = initial_strided_filter,
                                                  trainable = True)
            
            ou = output_shape[3]
            initial_strided_bias = np.zeros(ou).astype(np.float32)
            self.strided_bias = tf.get_variable('strided_bias',
                                                initializer = initial_strided_bias,
                                                trainable = True)
            
        self.parameters = [self.strided_filter, self.strided_bias]
        
        
    def StridedConvoltion(self,
            X,
            is_training
            ):
        #X = np.float32(X)

        '''
        NON TRIVIAL ROADBLOCK
        
        
        ok I have correct filter shape 
        now we are basically going back :) 
        
        suppose this 
        
        (N, 8, 8, 64) = conv(  (N, ?, ?, ?) | (5, 5, in, out))
        first shape is know since it is kinda passed to function
        second (output shape) shuold also be passed but lets calculate it
        
        
        '''
        
        conv_strided = tf.nn.conv2d_transpose(value = X,
                                              filter = self.strided_filter,
                                              output_shape = self.output_shape,
                                              strides = [1, self.strides, self.strides, 1],
                                              padding = 'SAME'
                                              )
        
        conv_strided = tf.nn.bias_add(conv_strided, self.strided_bias)
        if self.batch_norm:
            conv_strided = tf.contrib.layers.batch_norm(conv_strided,
                                                        decay = 0.9,
                                                        scale = True,
                                                        scope = self.name,
                                                        trainable = is_training,
                                                        is_training = is_training,
                                                        reuse = tf.AUTO_REUSE)
        
        return self.a(conv_strided)
    
    


#image = np.random.randn(10, 14, 14, 8).astype(np.float32)
#
##shape = ()
#init = np.random.randn(5,5,3,8).astype(np.float32)
#
#output_shape = (10, 28, 28, 3) 
#
#
#
#backwards_layer = FractionallyStriedConvLayer(filter_size = 5,
#                                              outmap = 3,
#                                              inmap = 8,
#                                              name = 'backconv1',
#                                              output_shape = output_shape,
#                                              strides = 2
#                                              )
#                                              
#conved = backwards_layer.StridedConvoltion(image, True)
#
#init = tf.global_variables_initializer()
#sess = tf.Session()
#
#sess.run(init)
#im = sess.run(conved)


# FINALLY GOT THAT BITCH WORKING THO
            
            
    
       
class ConnectedLayers():
    #calling it dense layer is for gays, who suck giant cocks 
    def __init__(self,
                 name,
                 shape,
                 activation = tf.nn.leaky_relu,
                 batch_norm = False):
        
        
        
        
        self.name = name
        self.shape = shape
        self.a = activation
        self.batch_norm = batch_norm
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            m, n = shape
            weight_initial = np.random.randn(m, n).astype(np.float32)
            self.weight = tf.get_variable('weight', 
                                          initializer = weight_initial,
                                          trainable = True
                                          )
            
            initial_bias = np.zeros(n).astype(np.float32)
            self.bias = tf.get_variable('bias',
                                        initializer = initial_bias,
                                        trainable = True)
    
        self.parameters = [self.weight, self.bias]
        
        
    def Forward(self,
                X,
                is_training = True,
                ):
        #X = np.float32(X)
        
        dotted = tf.matmul(X,self.weight) + self.bias
        
        if self.batch_norm:
            dotted = tf.contrib.layers.batch_norm(dotted,
                                                  decay = 0.9,
                                                  scale = True,
                                                  scope = self.name,
                                                  trainable = is_training,
                                                  is_training = is_training,
                                                  reuse = tf.AUTO_REUSE)
    
        return self.a(dotted)
    
    
    
#X = np.random.randn(10,5)
#
#layer = ConnectedLayers('layer1',
#                        (5,16))
#
#jela = layer.Forward(X, True)
#
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)
#dotted = sess.run(jela)








class DCGAN():
    
    def __init__(self,
                 image_dim,
                 color_channel,
                 D_CNN,
                 D_ANN,
                 batch_size,
                 G_CNN_TRANSPOSE,
                 G_ANN,
                 lr,
                 epoch,
                 output_activation,
                 latent_dimension = 100,
                 ):
        
        
        #if you think that this already seems fucked up
        #I agree with you 
        
        #also I cannot really debug it unless I finish that bad mother fucker 
        
        
        
        
        #GUESS I WILL JUST CREATE FUNCTIONS FIRST AND THEN GET BACK TO THIS LATER
        
        
        # CHAO AMIGO
        
        self.D_CNN = D_CNN
        self.D_ANN = D_ANN
        self.image_dim = image_dim
        self.color_channel = color_channel
        self.G_CNN_TRANSPOSE = G_CNN_TRANSPOSE #I will probabily have filters specified here
        self.G_ANN = G_ANN #probabily will have to remove this
        self.output_a = output_activation 
        self.latent_dims = latent_dimension 
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        #self.X = X
        # OK IM BACK 
        
        #LETS DO IT
        
        #...
        
        self.X_placeholder = tf.placeholder(tf.float32,name = 'tensorflow_X')
        self.Z = tf.placeholder(tf.float32, name = 'Tensorflow_Z')
        
        self.logits = self.Create_graph_disciminator_and_return_logits(X = self.X_placeholder,
                                                             #     D_CNN = self.D_CNN,
                                                           #       D_ANN = self.D_ANN)
                                                           )
        
        self.samples = self.Create_graph_generator_and_return_sampes(self.Z) #anu aq ideashi parametrebis micema araa aucilebeli

        #100% true mar ara jandaa unda aba : D : D vaaaaaa! 
        #radgan anu self-is attributebia ra   
        
        #welp...
        
        # au axla kai ragaca moxdeba :')

        logits_from_samples = self.Discriminator_FeedForward(X = self.samples,
                                                        is_training = True)
       

        #anu eseni arian logitebi migebuli generrebuli samplebisgan 
        
        #anu axla isev sampleebi unda gavchitot ogond test variantshi is batch_norm axurebs
        #test modeshi sxvanairia da trainigshi sxvanairi mara whatevaa ra
        
        self.test_samples = self.Generator_FeedBackward(self.Z, is_training = False)
        
        
        
        
        
        self.d_cost1 = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits,
                                                               labels = tf.ones_like(self.logits))
        
        self.d_cost2 = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_from_samples,
                                                               labels = tf.zeros_like(logits_from_samples))
        
        
        self.d_cost = tf.reduce_mean(self.d_cost1) + tf.reduce_mean(self.d_cost2)
        
        
        self.g_cost_pseudo = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_from_samples,
                                                                     labels = tf.ones_like(logits_from_samples))
        
        self.g_cost = tf.reduce_mean(self.g_cost_pseudo)
        
        real_pred = tf.cast(self.logits > 0, tf.float32)
        fake_pred = tf.cast(logits_from_samples < 0, tf.float32)
        num_preds = 2.0*self.batch_size
        correct_preds = tf.reduce_sum(real_pred) + tf.reduce_sum(fake_pred)
        self.accuracy = correct_preds/num_preds
        
        
        self.discriminator_parametes = [param for param in tf.trainable_variables() if param.name.startswith('D')]
        self.generator_parameters = [p for p in tf.trainable_variables() if p.name.startswith('G')]
        
        self.d_train_operation = tf.train.AdamOptimizer(self.lr).minimize(self.d_cost, var_list = self.discriminator_parametes)
        self.g_train_operation = tf.train.AdamOptimizer(self.lr).minimize(self.g_cost, var_list = self.generator_parameters)
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        return None 
    
    
        #if I get all these functions right im good to go
        #ok first function is to create graph 
        #second is to actually do stuff with them
        #now we do same with generator
        #and yeah that's basically it but generator feedbackward will be painfun
        #since it requires calcualting output shape at every fucking single node
    
    
    def Create_graph_disciminator_and_return_logits(self,
                                                    X
                                                    ):
        #anu jer am funqcias vidzaxebt da mogcems logitebs ra 
        #xoda mere sul ro agar ago es grafebi adgebi da forwards daudzaxeb pirdapir ra
        
        
        CNN_layers = self.D_CNN
        ANN_layers = self.D_ANN
        self.shape_np_array = (self.batch_size, self.image_dim, self.image_dim, self.color_channel)
        Dims = self.image_dim
        
        #ok I expect it to be a list of tuples just like always...
        
        #[(filter_shape), strides, batch_norm, activation)]
        
        i = 0;
        self.D_cnn_layers_array = []
        self.shape_copy = list(self.shape_np_array)
        for tuple_info_d in CNN_layers:
            shapee, strides, batch_boolean, activation =  tuple_info_d
            h, w, inm, outm = shapee
            what_the_fuck_is_going_ons = h, w, inm, outm
            
            
            self.shape_copy[1] = int(np.ceil(self.shape_copy[1]/strides))
            self.shape_copy[2] = int(np.ceil(self.shape_copy[2]/strides))
            self.shape_copy[3] = outm
            
            
            layer_conv_d = ConvLayer(name = 'Discriminator_conv_layer' + str(i),
                              shape_d = what_the_fuck_is_going_ons,
                              batch_norm = batch_boolean,
                              strides = strides,
                              activation = activation)
            i = i + 1;
            Dims = np.int32(np.ceil(np.float32(Dims)/ strides))
            
            self.D_cnn_layers_array.append(layer_conv_d)
            #now I should calculate dimension of flattened array : ) 

        #if it is a vector it should be noted here as well
        #real_first_dim = self.shape_copy[1]*self.shape_copy[2]*self.shape_copy[3]
        first_dim =  49 #shapee[3]*Dims*Dims #there we go : )
        
        i = 0; #now reset I and use it to initialize network with falttened dim vector
        
        # now we create regular layers 
        
        
        # I expect there to be (shape, batch_norm, activation)
        self.Discriminator_ann_layers = []
        last_tup = ANN_layers.pop(-1)
        for tuple_info in ANN_layers:

            shape, batch_boolean, activation = tuple_info
            
            if i == 0:
                m, n = shape
                shape_new = (first_dim, n)
                
                layer = ConnectedLayers(name = 'Discriminator_ann_layer' + str(i),
                                       shape = shape_new,
                                       activation = activation,
                                       batch_norm = batch_boolean)
                
            else:
                
                layer = ConnectedLayers(name = 'Discriminator_ann_layer' + str(i),
                                        shape = shape,
                                        activation = activation,
                                        batch_norm = batch_boolean)
                
        
            i += 1;
            
            self.Discriminator_ann_layers.append(layer)
            
        tup_info_last = last_tup
        shape, batch_boolean, activation = tup_info_last
        
        last_layer = ConnectedLayers(name = 'Discriminator_ann_layer' + str(i),
                                     shape = shape,
                                     activation = lambda x: x,
                                     batch_norm = batch_boolean)
        
        
        self.Discriminator_ann_layers.append(last_layer)
        
        #NOW WE HAVE BOTH GRAPHS AND HENCE CAN CALL FEEDFORWARD 
        
        
        
        
        
        logits = self.Discriminator_FeedForward(X, False)
        
        return logits 
    
    
    def Discriminator_FeedForward(self,
                                  X,
                                  is_training):
        
        output = X

        for JELA in self.D_cnn_layers_array:

            output = JELA.Convolution(output, is_training = is_training)

        output = tf.contrib.layers.flatten(output)

        for JEL in self.Discriminator_ann_layers:
            output = JEL.Forward(output, is_training = is_training)
        
        #output is actually a logit
        
        
        return output
    
    
            
    def Create_graph_generator_and_return_sampes(self,
                                                 Z):
        
        
        
        # main idea is that we know what output of a generator should be
        # meaning that we know what dimension we are looking for at output
        # since I want to feed network with filters
        # I will have to calcualte output shape of a network at every single node 
        
        # lets get started 
        
        '''
        image = np.random.randn(10, 14, 14, 8).astype(np.float32)
        
        init = np.random.randn(5,5,3,8).astype(np.float32)
        
        output_shape = (10, 28, 28, 3) 
        '''
        
        #G_ANN = [( 100, 200, batch_norm, activation)]
        #G_CNN = [ (  (filter_height, filter_width, output_channels, input_channels), strides, batch_norm, activation ) ... etc]
        #ANU REALURAD PIRDAPIR MECODINEBA ROGORI SHAPEPEBI UNDA MIVIGO DA TYNAURI AR MOMIWEVS MAGIS SHAPEEBIS DATVLAZE
        
        # ANU HO RA EGRE BEVRAD DALAGEBULI DA LAMAZI IQNEBA
        
        # PROSTA ERTI ES DENSE LAYEREBI GAVATYNA BOLOMDE DA EGA VAR
        
           
        
        
        #self.shape_np_array
        #output_shapes_at_all_stages.append(self.shape_np_array) #so first we start with image output
        
        
        N, dim1, dim2, colors = self.batch_size, self.image_dim, self.image_dim, self.color_channel
        output_shapes_at_all_stages = [(N, dim1, dim2, colors)]
        
        for layer_info in reversed(self.G_CNN_TRANSPOSE):
            shape, strides, batch_boolean, activation = layer_info
            height, width, output_maps, input_maps = shape
            dim1 = int(np.ceil(float(dim1)/strides))

            output_shape = (N, dim1, dim1, output_maps)

            output_shapes_at_all_stages.append(output_shape)
            
        #ok now output_shapes_at_all_stages is actually reversed of what is says : ) 
        
        #now we shoud create connected layers 
        
        # [ ( ( input, output), activation, apply_batch_norm )]
        
        self.G_ann_layers = []
        i = 0
        for layer_info in range(len(self.G_ANN) - 1):
            shape, batch_boolean, activation = self.G_ANN[layer_info]
            layer = ConnectedLayers('Generator_ann' + str(i),
                                    activation = activation,
                                    batch_norm = batch_boolean,
                                    shape = shape
                                    )
            
            
            
            self.G_ann_layers.append(layer)
            del layer
            i += 1;
        
        
        
        #now the last layer 
        
        last_layer_info = self.G_ANN[-1]
        
        shape, batch_boolean, activation = last_layer_info
        m, n = shape
        self.dim_backconv_first = output_shapes_at_all_stages[-1][2]
        self.n_prejection = n
        q = int(np.ceil(self.dim_backconv_first/strides))
        n = n*q*q # I will have to modify it to work on vectors as well
        self.n = n
        last_shape = (m, int(n))
        last_layer = ConnectedLayers('Generator_ann' + str(i),
                                     activation = activation,
                                     batch_norm = batch_boolean,
                                     shape = last_shape)
        
        self.G_ann_layers.append(last_layer)
                        
        #now we are forced to know input shape of back conv layer 
        
        self.G_backconv_layers = []
        
        #now reverse our good old output shapes list
        
        output_shapes_at_all_stages = list(reversed(output_shapes_at_all_stages))
        del i
        j = 0

        #[ ( (shape), strides, activation, batch_norm)]
        #anu pirvel layershi iqnebis ase; (5, 5, 10, 'x') da mere vtkvat (5, 5, 2, 10) output shapeebi ukve sheqmnili iqneba amis mixedvit!!!
       
        
        
        for layer_info in self.G_CNN_TRANSPOSE:
            
            shape, strides, batch_norm, activation = layer_info
            
            if j == 0:
                #this is da first mofo layer
                h, w, out, inn = shape 
                shape = h, w, out, self.n_prejection
                
            filt_sz_h, filt_sz_w, out, inmap = shape
            output_shape = output_shapes_at_all_stages[j]

            back_conv_layer = FractionallyStriedConvLayer(name = 'Generator_conv' + str(j),
                                                          outmap = out,
                                                          inmap = inmap,
                                                          filter_size = filt_sz_h,
                                                          output_shape = output_shape,
                                                          batch_norm = batch_boolean,
                                                          strides = strides,
                                                          activation = activation
                                                          )
                
            self.G_backconv_layers.append(back_conv_layer)
            j += 1
                     
            
        image_shape = output_shapes_at_all_stages[-1]
        outputt_map = image_shape[-1]
        input_map = out 
        

        last_conv_layer = FractionallyStriedConvLayer(name = 'Generator_conv' + str(j+1),
                                                      outmap = outputt_map,
                                                      inmap = input_map,
                                                      filter_size = 5,
                                                      output_shape = image_shape,
                                                      batch_norm = batch_boolean,
                                                      strides = strides,
                                                      activation = self.output_a
                                                      )


        self.G_backconv_layers.append(last_conv_layer)
            
        Z_samples = self.Generator_FeedBackward(Z, True)    
          
        return Z_samples
        
    
    def Generator_FeedBackward(self,
                               Z,
                               is_training):
        
        #after dense layers it should be transformed as a picture though 
        
        output = Z
        for layer in self.G_ann_layers:
            output = layer.Forward(output, is_training)
            
        #imaged = output.reshape(-1, self.dim_backconv_first, self.dim_backconv_first, self.n_prejection)
        dimm = int(np.ceil(self.dim_backconv_first/2))
        imaged = tf.reshape(output, [-1, dimm, dimm, self.n_prejection ])
               

        for layer_conv in self.G_backconv_layers:
            imaged = layer_conv.StridedConvoltion(imaged, is_training = is_training)

        return imaged
    
    def Fit(self, X):
        
        N = self.shape_np_array[0]
        shape = X.shape
        N = shape[0]
        N_batches = N // self.batch_size
        
        
        for i in range(self.epoch):
            np.random.shuffle(X)
            for j in range(N_batches):
                X_batches = X[j*self.batch_size: j*self.batch_size + self.batch_size]
                
                Z = np.random.uniform(-1, 1, size = (self.batch_size, self.latent_dims))
                
                _, D_accuracy, D_cost = self.sess.run((self.d_train_operation,
                                                      self.accuracy,
                                                      self.d_cost),
                                                      feed_dict = {self.X_placeholder: X_batches, self.Z: Z}
                                                      )        
        

                _, g_cost = self.sess.run((self.g_train_operation,
                                           self.g_cost),
                                           feed_dict = {self.Z: Z}
                                           )
                                           
                _, g_cost = self.sess.run((self.g_train_operation,
                                           self.g_cost),
                                           feed_dict = {self.Z: Z})
               
                _, g_cost = self.sess.run((self.g_train_operation,
                                           self.g_cost),
                                           feed_dict = {self.Z: Z})
        
                print('iteration ' + str(i) + ' batch ' + str(j))
                print('Accuracy of Discriminator ', end = '')
                print(D_accuracy)
                print('Discriminator Cost ', end = '')
                print(D_cost)
                print('Generator Cost ', end = ' ')
                print(g_cost)
                
        sampless = self.sess.run(self.samples, feed_dict = {self.Z: Z})
        print(sampless.shape)
        return sampless
    
    def Generate_Samples(self, n):
        Z = np.random.uniform(-1, 1, size = (n, self.latent_dims))
        samples = self.sess.run(self.test_samples, feed_dict = {self.Z: Z})
        
        return samples
        






































































