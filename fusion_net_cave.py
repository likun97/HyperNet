# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 11:35:47 2021

@author: lk
"""

#%%

import tensorflow as tf
import numpy as np
     
def compute_cost_l1_msssim(PredX , X_output): 
    loss_msssim  = 1-tf.reduce_mean(tf.image.ssim_multiscale(X_output, PredX, max_val=1.0, filter_size=6))  
    #loss_msssim = 1-tf.reduce_mean(tf.image.ssim_multiscale(X_output[0], PredX[0], max_val=1.0))  
    loss_l1      = tf.reduce_mean(tf.abs(PredX - X_output))
    return loss_l1,  loss_msssim  
  

def compute_cost( PredX , X_output): 
    cost = tf.reduce_mean(tf.square(PredX - X_output)) 
    return cost 
 

 
#%%

def fusion_net_MAE3(M_input, P_input , H_input):
    
    size_P = P_input.shape[1]
    size_M = M_input.shape[1] 
    size_H = H_input.shape[1] 
    
    diff_PM = size_P//size_M
    diff_PH = size_P//size_H
    diff_MH = size_M//size_H
 
    M_resize = tf.image.resize_images(M_input, size=[ M_input.shape[1]*diff_PM, M_input.shape[2]*diff_PM ], method=0)
    MP_mixed = tf.concat([M_resize, P_input] ,3)
     
    
    #################  Muiti-Scale-1 Convolution ##################################   
    with tf.variable_scope('layer1_MP_3x3'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        conv1_MP_3x3 = tf.nn.conv2d(MP_mixed, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_MP_3x3 = tf.nn.relu(conv1_MP_3x3)  
        
    with tf.variable_scope('layer1_MP_5x5'):
        weights      = tf.get_variable("w1_MP_3x3",[5,5,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        conv1_MP_5x5 = tf.nn.conv2d(MP_mixed, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_MP_5x5 = tf.nn.relu(conv1_MP_5x5)  
        
    with tf.variable_scope('layer1_MP_7x7'):
        weights      = tf.get_variable("w1_MP_3x3",[7,7,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        conv1_MP_7x7 = tf.nn.conv2d(MP_mixed, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_MP_7x7 = tf.nn.relu(conv1_MP_7x7)  
        
    conv1_MP_cat = tf.concat([conv1_MP_3x3,conv1_MP_5x5,conv1_MP_7x7],axis=-1)     
    
    print('conv1_MP_cat',conv1_MP_cat.shape)
    
    
    #################  Channel Attention 1 #######################################  
    CAttent_1_max  =  tf.reduce_max(conv1_MP_cat, axis=(1, 2), keepdims=True)      
    CAttent_1_mean = tf.reduce_mean(conv1_MP_cat, axis=(1, 2), keepdims=True)
    
    CAttent_1_maxm  =  CAttent_1_max + CAttent_1_mean                              
     
    print('CAttent_1_max',CAttent_1_max.shape)
    
    with tf.variable_scope('layer1_max_1'):
        weights   = tf.get_variable("w1_max_1",[1,1,48,24],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias      = tf.get_variable("b1_max_1",[24],initializer=tf.constant_initializer(0.0))
        conv1_max = tf.nn.conv2d(CAttent_1_maxm, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_max = tf.nn.relu(conv1_max)
        
    with tf.variable_scope('layer1_max_2'):
        weights     = tf.get_variable("w1_max_2",[1,1,24,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias        = tf.get_variable("b1max_2",[48],initializer=tf.constant_initializer(0.0))
        conv1_max_2 = tf.nn.conv2d(conv1_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        
    conv1_Catten_map = tf.nn.sigmoid(conv1_max_2)    
    conv1_Catten_out = conv1_MP_cat*conv1_Catten_map    
     
    print('conv1_Catten_out',conv1_Catten_out.shape)
    
    #################  Spatial Attention 1 #######################################  
    SAttent_1_max          = tf.reduce_max(conv1_Catten_out, axis=3, keepdims=True)
    SAttent_1_mean         = tf.reduce_mean(conv1_Catten_out, axis=3, keepdims=True)
    SAttent_1_cat_mean_max =  SAttent_1_max + SAttent_1_mean
    
    print('SAttent_1_cat_mean_max',SAttent_1_cat_mean_max.shape)                    
    
    with tf.variable_scope('layer1_atten_map'):
        weights  = tf.get_variable("w2_atten_map",[5,5,1,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
        conv1_Satten_map = tf.nn.conv2d(SAttent_1_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_Satten_map = tf.nn.sigmoid(conv1_Satten_map)  
        
    conv1_Satten_out = conv1_Catten_out*conv1_Satten_map    
   
    
    with tf.variable_scope('layer1_reduc1'):
        weights  = tf.get_variable("w",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[12],initializer=tf.constant_initializer(0.0))
        conv1_Satten_out = tf.nn.conv2d(conv1_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_Satten_out = tf.nn.relu(conv1_Satten_out) 

    with tf.variable_scope('layer1_reduc2'):
        weights  = tf.get_variable("w",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[4],initializer=tf.constant_initializer(0.0))
        conv1_Satten_out = tf.nn.conv2d(conv1_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_Satten_out = tf.nn.relu(conv1_Satten_out) 
        
    conv1_out = MP_mixed + conv1_Satten_out
    
    print('conv1_out',conv1_out.shape)                                             
    
    
    #################  Muiti-Scale-2 Convolution ################################# 
    # conv1_out     
    with tf.variable_scope('layer2_MP_3x3'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        conv2_MP_3x3 = tf.nn.conv2d(conv1_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_MP_3x3 = tf.nn.relu(conv2_MP_3x3)  
        
    with tf.variable_scope('layer2_MP_5x5'):
        weights      = tf.get_variable("w1_MP_5x5",[5,5,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_5x5",[16],initializer=tf.constant_initializer(0.0))  
        conv2_MP_5x5 = tf.nn.conv2d(conv1_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_MP_5x5 = tf.nn.relu(conv2_MP_5x5)  
        
    with tf.variable_scope('layer2_MP_7x7'):
        weights      = tf.get_variable("w1_MP_7x7",[7,7,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_7x7",[16],initializer=tf.constant_initializer(0.0))   
        conv2_MP_7x7 = tf.nn.conv2d(conv1_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_MP_7x7 = tf.nn.relu(conv2_MP_7x7)  
        
    conv2_MP_cat = tf.concat([conv2_MP_3x3,conv2_MP_5x5,conv2_MP_7x7],axis=-1)  
    print('conv2_MP_cat',conv2_MP_cat.shape)
    
    
    #################  Channel Attention 2 #######################################
    CAttent_2_max  =  tf.reduce_max(conv2_MP_cat, axis=(1, 2), keepdims=True)
    CAttent_2_mean = tf.reduce_mean(conv2_MP_cat, axis=(1, 2), keepdims=True)
    CAttent_2_maxm  =  CAttent_2_max + CAttent_2_mean
    print('CAttent_2_max',CAttent_2_max.shape)
    
    with tf.variable_scope('layer2_max_1'):
        weights   = tf.get_variable("w1_max_1",[1,1,48,24],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias      = tf.get_variable("b1_max_1",[24],initializer=tf.constant_initializer(0.0))
        conv2_max = tf.nn.conv2d(CAttent_2_maxm, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_max = tf.nn.relu(conv2_max)

    with tf.variable_scope('layer2_max_2'):
        weights     = tf.get_variable("w1_max_2",[1,1,24,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias        = tf.get_variable("b1max_2",[48],initializer=tf.constant_initializer(0.0))
        conv2_max_2 = tf.nn.conv2d(conv2_max, weights, strides=[1,1,1,1], padding='SAME') + bias
    
    conv2_Catten_map = tf.nn.sigmoid(conv2_max_2)    
    conv2_Catten_out = conv2_MP_cat*conv2_Catten_map    
     
    
    #################  Spatial Attention 2 #######################################  
    SAttent_2_max          = tf.reduce_max(conv2_Catten_out, axis=3, keepdims=True)
    SAttent_2_mean         = tf.reduce_mean(conv2_Catten_out, axis=3, keepdims=True)
    SAttent_2_cat_mean_max = SAttent_2_max + SAttent_2_mean
    
    with tf.variable_scope('layer2_atten_map'):
        weights  = tf.get_variable("w2_atten_map",[5,5,1,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
        conv2_Satten_map = tf.nn.conv2d(SAttent_2_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_Satten_map = tf.nn.sigmoid(conv2_Satten_map)  
        
    conv2_Satten_out = conv2_Catten_out*conv2_Satten_map    
 
    with tf.variable_scope('layer2_reduc1'):
        weights  = tf.get_variable("w",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[12],initializer=tf.constant_initializer(0.0))
        conv2_Satten_out = tf.nn.conv2d(conv2_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_Satten_out = tf.nn.relu(conv2_Satten_out) 

    with tf.variable_scope('layer2_reduc2'):
        weights  = tf.get_variable("w",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[4],initializer=tf.constant_initializer(0.0))
        conv2_Satten_out = tf.nn.conv2d(conv2_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_Satten_out = tf.nn.relu(conv2_Satten_out) 
        
    conv2_out = conv1_out + conv2_Satten_out
    print('conv2_out',conv2_out.shape)   
 
    
    ################  Muiti-Scale-3 Convolution #################################
    # conv2_out  
   
    with tf.variable_scope('layer3_MP_3x3'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv3_MP_3x3 = tf.nn.conv2d(conv2_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_MP_3x3 = tf.nn.relu(conv3_MP_3x3)  
        
    with tf.variable_scope('layer3_MP_5x5'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv3_MP_5x5 = tf.nn.conv2d(conv2_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_MP_5x5 = tf.nn.relu(conv3_MP_5x5)  
        
    with tf.variable_scope('layer3_MP_7x7'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv3_MP_7x7 = tf.nn.conv2d(conv2_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_MP_7x7 = tf.nn.relu(conv3_MP_7x7)  
        
    conv3_MP_cat = tf.concat([conv3_MP_3x3,conv3_MP_5x5,conv3_MP_7x7],axis=-1)  
    print('conv3_MP_cat',conv3_MP_cat.shape)
    
    
    #################  Channel Attention 3 #######################################
    CAttent_3_max  =  tf.reduce_max(conv3_MP_cat, axis=(1, 2), keepdims=True)
    CAttent_3_mean = tf.reduce_mean(conv3_MP_cat, axis=(1, 2), keepdims=True)
    CAttent_3_maxm = CAttent_3_max + CAttent_3_mean
    
    print('CAttent_3_max',CAttent_3_max.shape)
    
    with tf.variable_scope('layer3_max_1'):
        weights   = tf.get_variable("w1_max_1",[1,1,48,24],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias      = tf.get_variable("b1_max_1",[24],initializer=tf.constant_initializer(0.0))
        conv3_max = tf.nn.conv2d(CAttent_3_maxm, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_max = tf.nn.relu(conv3_max)
         
    with tf.variable_scope('layer3_max_2'):
        weights     = tf.get_variable("w1_max_2",[1,1,24,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias        = tf.get_variable("b1max_2",[48],initializer=tf.constant_initializer(0.0))
        conv3_max_2 = tf.nn.conv2d(conv3_max, weights, strides=[1,1,1,1], padding='SAME') + bias
    
    conv3_Catten_map = tf.nn.sigmoid(conv3_max_2)    
    conv3_Catten_out = conv3_MP_cat*conv3_Catten_map    
      
    
    #################  Spatial Attention 3 #######################################  
    SAttent_3_max          = tf.reduce_max(conv3_Catten_out, axis=3, keepdims=True)
    SAttent_3_mean         = tf.reduce_mean(conv3_Catten_out, axis=3, keepdims=True)
    SAttent_3_cat_mean_max = SAttent_3_max + SAttent_3_mean
    
    with tf.variable_scope('layer3_atten_map'):
        weights  = tf.get_variable("w2_atten_map",[5,5,1,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
        conv3_Satten_map = tf.nn.conv2d(SAttent_3_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_Satten_map = tf.nn.sigmoid(conv3_Satten_map)  
        
    conv3_Satten_out = conv3_Catten_out*conv3_Satten_map    
    
    with tf.variable_scope('layer3_reduc1'):
        weights  = tf.get_variable("w",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[12],initializer=tf.constant_initializer(0.0))
        conv3_Satten_out = tf.nn.conv2d(conv3_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_Satten_out = tf.nn.relu(conv3_Satten_out) 

    with tf.variable_scope('layer3_reduc2'):
        weights  = tf.get_variable("w",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[4],initializer=tf.constant_initializer(0.0))
        conv3_Satten_out = tf.nn.conv2d(conv3_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_Satten_out = tf.nn.relu(conv3_Satten_out) 
        
    conv3_out = conv2_out + conv3_Satten_out
    print('conv2_out',conv2_out.shape)   
     
     
    
    #========================================================================================================= 
    #################################################    Mh  &&  H     #######################################
    #=========================================================================================================   

    # H upsample 
    ######### 1 ######### 31.4
    # H_resise_orig          = tf.image.resize_images(H_input    , size=[ H_input.shape[1]*32 , H_input.shape[2]*32  ], method=0)
    ######### 2 ######### 31.7 
    H_middle     = tf.image.resize_images(images=H_input , size=[ H_input.shape[1]*diff_MH , H_input.shape[2]*diff_MH],method=tf.image.ResizeMethod.BICUBIC) 
    H_resize     = tf.image.resize_images(images=H_middle, size=[ H_input.shape[1]*diff_PH , H_input.shape[2]*diff_PH],method=tf.image.ResizeMethod.BICUBIC) 
    ######### 3 ######### 30.6
    # H_resise_orig          = tf.image.resize_images(H_input          , size=[ H_input.shape[1]*8 , H_input.shape[2]*8  ], method=0)
    # H_resise_orig          = tf.image.resize_images(H_resise_orig    , size=[ H_input.shape[1]*32 , H_input.shape[2]*32  ], method=0)
     
    # ================================
    kernel = np.ones((5,5))/25.0      
    kernel = np.expand_dims( kernel ,-1)
    kernel = np.expand_dims( kernel ,-1)
    
    kernel1 = np.repeat(kernel , 4, axis=2)      
    kernel3 = np.repeat(kernel ,31, axis=2)
 
    Kernel_MP = tf.get_variable ( 'Mp' ,[5, 5, 4, 1] , tf.float32, initializer=tf.constant_initializer(kernel1) )    
    Kernel_H  = tf.get_variable ( 'H' ,[5, 5, 31,1], tf.float32 , initializer=tf.constant_initializer(kernel3) )  
     
    MPF_h   = tf.nn.depthwise_conv2d( conv3_out, Kernel_MP, strides=[1,1,1,1], padding='SAME')            
    H_h   = tf.nn.depthwise_conv2d( H_resize, Kernel_H, strides=[1,1,1,1], padding='SAME')          
      
    MPF_filter  = conv3_out 
    H_filter    = H_resize  
      
    ######################## H Layer 1  
    with tf.variable_scope('layer1_H'):
        weights  = tf.get_variable("w1",[3,3,31,31],initializer=tf.truncated_normal_initializer(stddev=1e-3))        
        bias     = tf.get_variable("b1",[31],initializer=tf.constant_initializer(0.0))
        conv1_H  = tf.nn.conv2d(H_filter, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_H  = tf.nn.relu(conv1_H)    # 31

    ################ MPF Layer 1     
    with tf.variable_scope('layer1_MPF'):
        weights   = tf.get_variable("w1",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias      = tf.get_variable("b1",[16],initializer=tf.constant_initializer(0.0))
        conv1_MPF = tf.nn.conv2d(MPF_filter, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_MPF = tf.nn.relu(conv1_MPF)   # 16 

 
    ######################## H Layer 2  
    H_dense_2 = tf.concat([H_filter,conv1_H,conv1_MPF],axis=-1) # 78
    
    with tf.variable_scope('layer2_H'):
        weights  = tf.get_variable("w2",[3,3,78,31],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias     = tf.get_variable("b2",[31],initializer=tf.constant_initializer(0.0))
        conv2_H  = tf.nn.conv2d(H_dense_2, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_H  = tf.nn.relu(conv2_H)    # 31
        
    ################ MPF Layer 2     
    MPF_dense_2 = tf.concat([MPF_filter,conv1_MPF],axis=-1)  # 20
    
    with tf.variable_scope('layer2_MPF'):
        weights   = tf.get_variable("w2",[3,3,20,8],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias      = tf.get_variable("b2",[8],initializer=tf.constant_initializer(0.0))
        conv2_MPF = tf.nn.conv2d(MPF_dense_2, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_MPF = tf.nn.relu(conv2_MPF)    # 8
    
    ######################## H Layer 3  
    H_dense_3 = tf.concat([H_filter, conv1_H, conv2_H, conv2_MPF],axis=-1) # 101
    
    with tf.variable_scope('layer3_H'):
        weights  = tf.get_variable("w3",[3,3,101,31],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias     = tf.get_variable("b3",[31],initializer=tf.constant_initializer(0.0))
        conv3_H  = tf.nn.conv2d(H_dense_3, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_H  = tf.nn.relu(conv3_H)  # 31
        
        ################ MPF Layer 3     
    MPF_dense_3 = tf.concat([MPF_filter,conv1_MPF,conv2_MPF],axis=-1) # 28
    
    with tf.variable_scope('layer3_MPF'):
        weights   = tf.get_variable("w3",[3,3,28,8],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias      = tf.get_variable("b3",[8],initializer=tf.constant_initializer(0.0))
        conv3_MPF = tf.nn.conv2d(MPF_dense_3, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_MPF = tf.nn.relu(conv3_MPF)    # 8 
    
    
    ######################## H Layer 4  
    H_dense_4 = tf.concat([H_filter, conv1_H, conv2_H, conv3_H, conv3_MPF],axis=-1) # 31*5+24 159   132
    
    with tf.variable_scope('layer4_H'):
        weights  = tf.get_variable("w4",[3,3,132,68],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias     = tf.get_variable("b4",[68],initializer=tf.constant_initializer(0.0))
        conv4_H  = tf.nn.conv2d(H_dense_4, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_H  = tf.nn.relu(conv4_H) 
        
        weights  = tf.get_variable("w4_",[3,3,68,31],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias     = tf.get_variable("b4_",[31],initializer=tf.constant_initializer(0.0))
        conv4_H  = tf.nn.conv2d(conv4_H, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_H  = tf.nn.relu(conv4_H) 
 
    mix_H_add = conv4_H   
    print('mix_H_add',mix_H_add.shape) 
    return mix_H_add



#%%

def fusion_net_MAE6(M_input, P_input , H_input):
    
    size_P = P_input.shape[1]
    size_M = M_input.shape[1] 
    size_H = H_input.shape[1] 
    
    diff_PM = size_P//size_M
    diff_PH = size_P//size_H
    diff_MH = size_M//size_H
 
    M_resize = tf.image.resize_images(M_input, size=[ M_input.shape[1]*diff_PM, M_input.shape[2]*diff_PM ], method=0)
    
    MP_mixed = tf.concat([M_resize, P_input] ,3)
    
      
    ############################################################################# 
    #################  Muiti-Scale-1 Convolution ##################################   

    with tf.variable_scope('layer1_MP_3x3'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        conv1_MP_3x3 = tf.nn.conv2d(MP_mixed, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_MP_3x3 = tf.nn.relu(conv1_MP_3x3)  
        
    with tf.variable_scope('layer1_MP_5x5'):
        weights      = tf.get_variable("w1_MP_3x3",[5,5,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        conv1_MP_5x5 = tf.nn.conv2d(MP_mixed, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_MP_5x5 = tf.nn.relu(conv1_MP_5x5)  
        
    with tf.variable_scope('layer1_MP_7x7'):
        weights      = tf.get_variable("w1_MP_3x3",[7,7,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        conv1_MP_7x7 = tf.nn.conv2d(MP_mixed, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_MP_7x7 = tf.nn.relu(conv1_MP_7x7)  
        
    conv1_MP_cat = tf.concat([conv1_MP_3x3,conv1_MP_5x5,conv1_MP_7x7],axis=-1)     # (12, 256, 256, 48)
    
    print('conv1_MP_cat',conv1_MP_cat.shape)
    
    
    #################  Channel Attention 1 #######################################  
  
    CAttent_1_max  =  tf.reduce_max(conv1_MP_cat, axis=(1, 2), keepdims=True)      # (12, 1  , 1  , 48)
    CAttent_1_mean = tf.reduce_mean(conv1_MP_cat, axis=(1, 2), keepdims=True)
    
    CAttent_1_maxm  =  CAttent_1_max + CAttent_1_mean                              # (12, 1  , 1  , 48) 
     
    print('CAttent_1_max',CAttent_1_max.shape)
    
    with tf.variable_scope('layer1_max_1'):
        weights   = tf.get_variable("w1_max_1",[1,1,48,24],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias      = tf.get_variable("b1_max_1",[24],initializer=tf.constant_initializer(0.0))
        conv1_max = tf.nn.conv2d(CAttent_1_maxm, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_max = tf.nn.relu(conv1_max)
        
    with tf.variable_scope('layer1_max_2'):
        weights     = tf.get_variable("w1_max_2",[1,1,24,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias        = tf.get_variable("b1max_2",[48],initializer=tf.constant_initializer(0.0))
        conv1_max_2 = tf.nn.conv2d(conv1_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        
    conv1_Catten_map = tf.nn.sigmoid(conv1_max_2)    
    conv1_Catten_out = conv1_MP_cat*conv1_Catten_map    
     
    print('conv1_Catten_out',conv1_Catten_out.shape)
    
    #################  Spatial Attention 1 #######################################  

    SAttent_1_max          = tf.reduce_max(conv1_Catten_out, axis=3, keepdims=True)
    SAttent_1_mean         = tf.reduce_mean(conv1_Catten_out, axis=3, keepdims=True)
    SAttent_1_cat_mean_max =  SAttent_1_max + SAttent_1_mean
    
    print('SAttent_1_cat_mean_max',SAttent_1_cat_mean_max.shape)                    #  (12, 256, 256, 2)
    
    with tf.variable_scope('layer1_atten_map'):
        weights  = tf.get_variable("w2_atten_map",[5,5,1,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
        conv1_Satten_map = tf.nn.conv2d(SAttent_1_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_Satten_map = tf.nn.sigmoid(conv1_Satten_map)  
        
    conv1_Satten_out = conv1_Catten_out*conv1_Satten_map    
   
    
    with tf.variable_scope('layer1_reduc1'):
        weights  = tf.get_variable("w",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[12],initializer=tf.constant_initializer(0.0))
        conv1_Satten_out = tf.nn.conv2d(conv1_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_Satten_out = tf.nn.relu(conv1_Satten_out) 

    with tf.variable_scope('layer1_reduc2'):
        weights  = tf.get_variable("w",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[4],initializer=tf.constant_initializer(0.0))
        conv1_Satten_out = tf.nn.conv2d(conv1_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_Satten_out = tf.nn.relu(conv1_Satten_out) 
        
    conv1_out = MP_mixed + conv1_Satten_out 
    print('conv1_out',conv1_out.shape)                                              #  (12, 256, 256, 48)
  

    ############################################################################# 
    #################  Muiti-Scale-2 Convolution ################################# 
    # conv1_out    
   
    with tf.variable_scope('layer2_MP_3x3'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        conv2_MP_3x3 = tf.nn.conv2d(conv1_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_MP_3x3 = tf.nn.relu(conv2_MP_3x3)  
        
    with tf.variable_scope('layer2_MP_5x5'):
        weights      = tf.get_variable("w1_MP_5x5",[5,5,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_5x5",[16],initializer=tf.constant_initializer(0.0))  
        conv2_MP_5x5 = tf.nn.conv2d(conv1_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_MP_5x5 = tf.nn.relu(conv2_MP_5x5)  
        
    with tf.variable_scope('layer2_MP_7x7'):
        weights      = tf.get_variable("w1_MP_7x7",[7,7,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_7x7",[16],initializer=tf.constant_initializer(0.0))   
        conv2_MP_7x7 = tf.nn.conv2d(conv1_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_MP_7x7 = tf.nn.relu(conv2_MP_7x7)  
        
    conv2_MP_cat = tf.concat([conv2_MP_3x3,conv2_MP_5x5,conv2_MP_7x7],axis=-1)   
    print('conv2_MP_cat',conv2_MP_cat.shape)
     
    #################  Channel Attention 2 ####################################### 
    CAttent_2_max  =  tf.reduce_max(conv2_MP_cat, axis=(1, 2), keepdims=True)
    CAttent_2_mean = tf.reduce_mean(conv2_MP_cat, axis=(1, 2), keepdims=True) 
    CAttent_2_maxm  =  CAttent_2_max + CAttent_2_mean
    
    print('CAttent_2_max',CAttent_2_max.shape)
    
    with tf.variable_scope('layer2_max_1'):
        weights   = tf.get_variable("w1_max_1",[1,1,48,24],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias      = tf.get_variable("b1_max_1",[24],initializer=tf.constant_initializer(0.0))
        conv2_max = tf.nn.conv2d(CAttent_2_maxm, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_max = tf.nn.relu(conv2_max)
            

    with tf.variable_scope('layer2_max_2'):
        weights     = tf.get_variable("w1_max_2",[1,1,24,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias        = tf.get_variable("b1max_2",[48],initializer=tf.constant_initializer(0.0))
        conv2_max_2 = tf.nn.conv2d(conv2_max, weights, strides=[1,1,1,1], padding='SAME') + bias
    
    conv2_Catten_map = tf.nn.sigmoid(conv2_max_2)    
    conv2_Catten_out = conv2_MP_cat*conv2_Catten_map    
     
    
    #################  Spatial Attention 2 #######################################  

    SAttent_2_max          = tf.reduce_max(conv2_Catten_out, axis=3, keepdims=True)
    SAttent_2_mean         = tf.reduce_mean(conv2_Catten_out, axis=3, keepdims=True)
    SAttent_2_cat_mean_max = SAttent_2_max + SAttent_2_mean
    
    with tf.variable_scope('layer2_atten_map'):
        weights  = tf.get_variable("w2_atten_map",[5,5,1,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
        conv2_Satten_map = tf.nn.conv2d(SAttent_2_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_Satten_map = tf.nn.sigmoid(conv2_Satten_map)  
        
    conv2_Satten_out = conv2_Catten_out*conv2_Satten_map    
     
    with tf.variable_scope('layer2_reduc1'):
        weights  = tf.get_variable("w",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[12],initializer=tf.constant_initializer(0.0))
        conv2_Satten_out = tf.nn.conv2d(conv2_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_Satten_out = tf.nn.relu(conv2_Satten_out) 

    with tf.variable_scope('layer2_reduc2'):
        weights  = tf.get_variable("w",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[4],initializer=tf.constant_initializer(0.0))
        conv2_Satten_out = tf.nn.conv2d(conv2_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_Satten_out = tf.nn.relu(conv2_Satten_out) 
        
    conv2_out = conv1_out + conv2_Satten_out
    
    print('conv2_out',conv2_out.shape)   
 

    ############################################################################# 
    ################  Muiti-Scale-3 Convolution #################################

    # conv2_out    
    with tf.variable_scope('layer3_MP_3x3'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv3_MP_3x3 = tf.nn.conv2d(conv2_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_MP_3x3 = tf.nn.relu(conv3_MP_3x3)  
        
    with tf.variable_scope('layer3_MP_5x5'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv3_MP_5x5 = tf.nn.conv2d(conv2_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_MP_5x5 = tf.nn.relu(conv3_MP_5x5)  
        
    with tf.variable_scope('layer3_MP_7x7'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv3_MP_7x7 = tf.nn.conv2d(conv2_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_MP_7x7 = tf.nn.relu(conv3_MP_7x7)  
        
    conv3_MP_cat = tf.concat([conv3_MP_3x3,conv3_MP_5x5,conv3_MP_7x7],axis=-1)  
    
    print('conv3_MP_cat',conv3_MP_cat.shape)
        
    #################  Channel Attention 3 #######################################
    
    CAttent_3_max  =  tf.reduce_max(conv3_MP_cat, axis=(1, 2), keepdims=True)
    CAttent_3_mean = tf.reduce_mean(conv3_MP_cat, axis=(1, 2), keepdims=True)
    
    CAttent_3_maxm = CAttent_3_max + CAttent_3_mean
    
    print('CAttent_3_max',CAttent_3_max.shape)
    
    with tf.variable_scope('layer3_max_1'):
        weights   = tf.get_variable("w1_max_1",[1,1,48,24],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias      = tf.get_variable("b1_max_1",[24],initializer=tf.constant_initializer(0.0))
        conv3_max = tf.nn.conv2d(CAttent_3_maxm, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_max = tf.nn.relu(conv3_max)
         

    with tf.variable_scope('layer3_max_2'):
        weights     = tf.get_variable("w1_max_2",[1,1,24,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias        = tf.get_variable("b1max_2",[48],initializer=tf.constant_initializer(0.0))
        conv3_max_2 = tf.nn.conv2d(conv3_max, weights, strides=[1,1,1,1], padding='SAME') + bias
    
    conv3_Catten_map = tf.nn.sigmoid(conv3_max_2)    
    conv3_Catten_out = conv3_MP_cat*conv3_Catten_map    
          
    #################  Spatial Attention 3 #######################################  

    SAttent_3_max          = tf.reduce_max(conv3_Catten_out, axis=3, keepdims=True)
    SAttent_3_mean         = tf.reduce_mean(conv3_Catten_out, axis=3, keepdims=True)
    SAttent_3_cat_mean_max = SAttent_3_max + SAttent_3_mean
    
    with tf.variable_scope('layer3_atten_map'):
        weights  = tf.get_variable("w2_atten_map",[5,5,1,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
        conv3_Satten_map = tf.nn.conv2d(SAttent_3_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_Satten_map = tf.nn.sigmoid(conv3_Satten_map)  
        
    conv3_Satten_out = conv3_Catten_out*conv3_Satten_map    
    
    with tf.variable_scope('layer3_reduc1'):
        weights  = tf.get_variable("w",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[12],initializer=tf.constant_initializer(0.0))
        conv3_Satten_out = tf.nn.conv2d(conv3_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_Satten_out = tf.nn.relu(conv3_Satten_out) 

    with tf.variable_scope('layer3_reduc2'):
        weights  = tf.get_variable("w",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[4],initializer=tf.constant_initializer(0.0))
        conv3_Satten_out = tf.nn.conv2d(conv3_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_Satten_out = tf.nn.relu(conv3_Satten_out) 
        
    conv3_out = conv2_out + conv3_Satten_out
    
    print('conv2_out',conv2_out.shape)   
        
    ############################################################################# 
    ################  Muiti-Scale-4 Convolution #################################

    # conv3_out    
   
    with tf.variable_scope('layer4_MP_3x3'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv4_MP_3x3 = tf.nn.conv2d(conv3_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_MP_3x3 = tf.nn.relu(conv4_MP_3x3)  
        
    with tf.variable_scope('layer4_MP_5x5'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv4_MP_5x5 = tf.nn.conv2d(conv3_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_MP_5x5 = tf.nn.relu(conv4_MP_5x5)  
        
    with tf.variable_scope('layer4_MP_7x7'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv4_MP_7x7 = tf.nn.conv2d(conv3_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_MP_7x7 = tf.nn.relu(conv4_MP_7x7)  
        
    conv4_MP_cat = tf.concat([conv4_MP_3x3,conv4_MP_5x5,conv4_MP_7x7],axis=-1)  
    
    print('conv4_MP_cat',conv4_MP_cat.shape)
    
    #################  Channel Attention 3 #######################################
        
    CAttent_4_max  =  tf.reduce_max(conv4_MP_cat, axis=(1, 2), keepdims=True)
    CAttent_4_mean = tf.reduce_mean(conv4_MP_cat, axis=(1, 2), keepdims=True)
    
    CAttent_4_maxm = CAttent_4_max + CAttent_4_mean
    
    print('CAttent_4_max',CAttent_4_max.shape)
    
    with tf.variable_scope('layer4_max_1'):
        weights   = tf.get_variable("w1_max_1",[1,1,48,24],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias      = tf.get_variable("b1_max_1",[24],initializer=tf.constant_initializer(0.0))
        conv4_max = tf.nn.conv2d(CAttent_4_maxm, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_max = tf.nn.relu(conv4_max)
         

    with tf.variable_scope('layer4_max_2'):
        weights     = tf.get_variable("w1_max_2",[1,1,24,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias        = tf.get_variable("b1max_2",[48],initializer=tf.constant_initializer(0.0))
        conv4_max_2 = tf.nn.conv2d(conv4_max, weights, strides=[1,1,1,1], padding='SAME') + bias
    
    conv4_Catten_map = tf.nn.sigmoid(conv4_max_2)    
    conv4_Catten_out = conv4_MP_cat*conv4_Catten_map    
      
    #################  Spatial Attention 3 #######################################  

    SAttent_4_max          = tf.reduce_max(conv4_Catten_out, axis=3, keepdims=True)
    SAttent_4_mean         = tf.reduce_mean(conv4_Catten_out, axis=3, keepdims=True)
    SAttent_4_cat_mean_max = SAttent_4_max + SAttent_4_mean
    
    with tf.variable_scope('layer4_atten_map'):
        weights  = tf.get_variable("w2_atten_map",[5,5,1,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
        conv4_Satten_map = tf.nn.conv2d(SAttent_4_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_Satten_map = tf.nn.sigmoid(conv4_Satten_map)  
        
    conv4_Satten_out = conv4_Catten_out*conv4_Satten_map    
     
    with tf.variable_scope('layer4_reduc1'):
        weights  = tf.get_variable("w",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[12],initializer=tf.constant_initializer(0.0))
        conv4_Satten_out = tf.nn.conv2d(conv4_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_Satten_out = tf.nn.relu(conv4_Satten_out) 

    with tf.variable_scope('layer4_reduc2'):
        weights  = tf.get_variable("w",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[4],initializer=tf.constant_initializer(0.0))
        conv4_Satten_out = tf.nn.conv2d(conv4_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_Satten_out = tf.nn.relu(conv4_Satten_out) 
        
    conv4_out = conv3_out + conv4_Satten_out
    
    print('conv3_out',conv3_out.shape)   
    
      
    ############################################################################# 
    ################  Muiti-Scale-5 Convolution #################################

    # conv4_out    
   
    with tf.variable_scope('layer5_MP_3x3'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv5_MP_3x3 = tf.nn.conv2d(conv4_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv5_MP_3x3 = tf.nn.relu(conv5_MP_3x3)  
        
    with tf.variable_scope('layer5_MP_5x5'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv5_MP_5x5 = tf.nn.conv2d(conv4_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv5_MP_5x5 = tf.nn.relu(conv5_MP_5x5)  
        
    with tf.variable_scope('layer5_MP_7x7'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv5_MP_7x7 = tf.nn.conv2d(conv4_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv5_MP_7x7 = tf.nn.relu(conv5_MP_7x7)  
        
    conv5_MP_cat = tf.concat([conv5_MP_3x3,conv5_MP_5x5,conv5_MP_7x7],axis=-1)  
    
    print('conv5_MP_cat',conv5_MP_cat.shape)
        
    #################  Channel Attention 3 #######################################
        
    CAttent_5_max  =  tf.reduce_max(conv5_MP_cat, axis=(1, 2), keepdims=True)
    CAttent_5_mean = tf.reduce_mean(conv5_MP_cat, axis=(1, 2), keepdims=True)
    
    CAttent_5_maxm = CAttent_5_max + CAttent_5_mean
    
    print('CAttent_5_max',CAttent_5_max.shape)
    
    with tf.variable_scope('layer5_max_1'):
        weights   = tf.get_variable("w1_max_1",[1,1,48,24],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias      = tf.get_variable("b1_max_1",[24],initializer=tf.constant_initializer(0.0))
        conv5_max = tf.nn.conv2d(CAttent_5_maxm, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv5_max = tf.nn.relu(conv5_max)
         

    with tf.variable_scope('layer5_max_2'):
        weights     = tf.get_variable("w1_max_2",[1,1,24,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias        = tf.get_variable("b1max_2",[48],initializer=tf.constant_initializer(0.0))
        conv5_max_2 = tf.nn.conv2d(conv5_max, weights, strides=[1,1,1,1], padding='SAME') + bias
    
    conv5_Catten_map = tf.nn.sigmoid(conv5_max_2)    
    conv5_Catten_out = conv5_MP_cat*conv5_Catten_map    
          
    #################  Spatial Attention 3 #######################################  

    SAttent_5_max          = tf.reduce_max(conv5_Catten_out, axis=3, keepdims=True)
    SAttent_5_mean         = tf.reduce_mean(conv5_Catten_out, axis=3, keepdims=True)
    SAttent_5_cat_mean_max = SAttent_5_max + SAttent_5_mean
    
    with tf.variable_scope('layer5_atten_map'):
        weights  = tf.get_variable("w2_atten_map",[5,5,1,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
        conv5_Satten_map = tf.nn.conv2d(SAttent_5_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv5_Satten_map = tf.nn.sigmoid(conv5_Satten_map)  
        
    conv5_Satten_out = conv5_Catten_out*conv5_Satten_map    
      
    with tf.variable_scope('layer5_reduc1'):
        weights  = tf.get_variable("w",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[12],initializer=tf.constant_initializer(0.0))
        conv5_Satten_out = tf.nn.conv2d(conv5_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv5_Satten_out = tf.nn.relu(conv5_Satten_out) 

    with tf.variable_scope('layer5_reduc2'):
        weights  = tf.get_variable("w",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[4],initializer=tf.constant_initializer(0.0))
        conv5_Satten_out = tf.nn.conv2d(conv5_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv5_Satten_out = tf.nn.relu(conv5_Satten_out) 
        
    conv5_out = conv4_out + conv5_Satten_out
    
    print('conv4_out',conv4_out.shape)   
    
    
    ############################################################################# 
    ################  Muiti-Scale-6 Convolution #################################

    # conv4_out     
   
    with tf.variable_scope('layer6_MP_3x3'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv6_MP_3x3 = tf.nn.conv2d(conv5_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv6_MP_3x3 = tf.nn.relu(conv6_MP_3x3)  
        
    with tf.variable_scope('layer6_MP_5x5'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv6_MP_5x5 = tf.nn.conv2d(conv5_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv6_MP_5x5 = tf.nn.relu(conv6_MP_5x5)  
        
    with tf.variable_scope('layer6_MP_7x7'):
        weights      = tf.get_variable("w1_MP_3x3",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias         = tf.get_variable("b1_MP_3x3",[16],initializer=tf.constant_initializer(0.0))
        
        conv6_MP_7x7 = tf.nn.conv2d(conv5_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv6_MP_7x7 = tf.nn.relu(conv6_MP_7x7)  
        
    conv6_MP_cat = tf.concat([conv6_MP_3x3,conv6_MP_5x5,conv6_MP_7x7],axis=-1)  
    
    print('conv6_MP_cat',conv6_MP_cat.shape)
        
    #################  Channel Attention 3 #######################################
    
    CAttent_6_max  =  tf.reduce_max(conv6_MP_cat, axis=(1, 2), keepdims=True)
    CAttent_6_mean = tf.reduce_mean(conv6_MP_cat, axis=(1, 2), keepdims=True)
    
    CAttent_6_maxm = CAttent_6_max + CAttent_6_mean
    
    print('CAttent_6_max',CAttent_6_max.shape)
    
    with tf.variable_scope('layer6_max_1'):
        weights   = tf.get_variable("w1_max_1",[1,1,48,24],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias      = tf.get_variable("b1_max_1",[24],initializer=tf.constant_initializer(0.0))
        conv6_max = tf.nn.conv2d(CAttent_6_maxm, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv6_max = tf.nn.relu(conv6_max)
         

    with tf.variable_scope('layer6_max_2'):
        weights     = tf.get_variable("w1_max_2",[1,1,24,48],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias        = tf.get_variable("b1max_2",[48],initializer=tf.constant_initializer(0.0))
        conv6_max_2 = tf.nn.conv2d(conv6_max, weights, strides=[1,1,1,1], padding='SAME') + bias
    
    conv6_Catten_map = tf.nn.sigmoid(conv6_max_2)    
    conv6_Catten_out = conv6_MP_cat*conv6_Catten_map    
      
    #################  Spatial Attention 3 #######################################  

    SAttent_6_max          = tf.reduce_max(conv6_Catten_out, axis=3, keepdims=True)
    SAttent_6_mean         = tf.reduce_mean(conv6_Catten_out, axis=3, keepdims=True)
    SAttent_6_cat_mean_max = SAttent_6_max + SAttent_6_mean
    
    with tf.variable_scope('layer6_atten_map'):
        weights  = tf.get_variable("w2_atten_map",[5,5,1,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
        conv6_Satten_map = tf.nn.conv2d(SAttent_6_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv6_Satten_map = tf.nn.sigmoid(conv6_Satten_map)  
        
    conv6_Satten_out = conv6_Catten_out*conv6_Satten_map    
    
  
    with tf.variable_scope('layer6_reduc1'):
        weights  = tf.get_variable("w",[3,3,48,12],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[12],initializer=tf.constant_initializer(0.0))
        conv6_Satten_out = tf.nn.conv2d(conv6_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv6_Satten_out = tf.nn.relu(conv6_Satten_out) 

    with tf.variable_scope('layer6_reduc2'):
        weights  = tf.get_variable("w",[3,3,12,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias     = tf.get_variable("b",[4],initializer=tf.constant_initializer(0.0))
        conv6_Satten_out = tf.nn.conv2d(conv6_Satten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv6_Satten_out = tf.nn.relu(conv6_Satten_out) 
        
    conv6_out = conv5_out + conv6_Satten_out
    
    print('conv5_out',conv5_out.shape)    
    #========================================================================================================= 
    #################################################    Mh  &&  H     #######################################
    #=========================================================================================================   

    # H 上采样 
    ######### 1 ######### 31.4
    # H_resise_orig          = tf.image.resize_images(H_input    , size=[ H_input.shape[1]*32 , H_input.shape[2]*32  ], method=0)
    ######### 2 ######### 31.7 
    H_middle     = tf.image.resize_images(images=H_input , size=[ H_input.shape[1]*diff_MH , H_input.shape[2]*diff_MH],method=tf.image.ResizeMethod.BICUBIC) 
    H_resize     = tf.image.resize_images(images=H_middle, size=[ H_input.shape[1]*diff_PH , H_input.shape[2]*diff_PH],method=tf.image.ResizeMethod.BICUBIC) 
    ######### 3 ######### 30.6
    # H_resise_orig          = tf.image.resize_images(H_input          , size=[ H_input.shape[1]*8 , H_input.shape[2]*8  ], method=0)
    # H_resise_orig          = tf.image.resize_images(H_resise_orig    , size=[ H_input.shape[1]*32 , H_input.shape[2]*32  ], method=0)
     
    kernel = np.ones((5,5))/25.0      
    kernel = np.expand_dims( kernel ,-1)
    kernel = np.expand_dims( kernel ,-1)
    
    kernel1 = np.repeat(kernel , 4, axis=2)      
    kernel3 = np.repeat(kernel ,31, axis=2)
 
    Kernel_MP = tf.get_variable ( 'Mp' ,[5, 5, 4, 1] , tf.float32, initializer=tf.constant_initializer(kernel1) )    
    Kernel_H  = tf.get_variable ( 'H' ,[5, 5, 31,1], tf.float32 , initializer=tf.constant_initializer(kernel3) )  
     
    MPF_h   = tf.nn.depthwise_conv2d( conv6_out, Kernel_MP, strides=[1,1,1,1], padding='SAME')            
    H_h   = tf.nn.depthwise_conv2d( H_resize, Kernel_H, strides=[1,1,1,1], padding='SAME')          
      
    MPF_filter  = conv6_out 
    H_filter    = H_resize  
     
         
    ######################## H Layer 1  
    with tf.variable_scope('layer1_H'):
        weights  = tf.get_variable("w1",[3,3,31,31],initializer=tf.truncated_normal_initializer(stddev=1e-3))             
        bias     = tf.get_variable("b1",[31],initializer=tf.constant_initializer(0.0))
        conv1_H  = tf.nn.conv2d(H_filter, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_H  = tf.nn.relu(conv1_H)    # 31

        ################ MPF Layer 1     
    with tf.variable_scope('layer1_MPF'):
        weights   = tf.get_variable("w1",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias      = tf.get_variable("b1",[16],initializer=tf.constant_initializer(0.0))
        conv1_MPF = tf.nn.conv2d(MPF_filter, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv1_MPF = tf.nn.relu(conv1_MPF)   # 16 

    
    ######################## H Layer 2  
    H_dense_2 = tf.concat([H_filter,conv1_H,conv1_MPF],axis=-1) # 78
    
    with tf.variable_scope('layer2_H'):
        weights  = tf.get_variable("w2",[3,3,78,31],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias     = tf.get_variable("b2",[31],initializer=tf.constant_initializer(0.0))
        conv2_H  = tf.nn.conv2d(H_dense_2, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_H  = tf.nn.relu(conv2_H)    # 31
        
        ################ MPF Layer 2     
    MPF_dense_2 = tf.concat([MPF_filter,conv1_MPF],axis=-1)  # 20
    
    with tf.variable_scope('layer2_MPF'):
        weights   = tf.get_variable("w2",[3,3,20,8],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias      = tf.get_variable("b2",[8],initializer=tf.constant_initializer(0.0))
        conv2_MPF = tf.nn.conv2d(MPF_dense_2, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv2_MPF = tf.nn.relu(conv2_MPF)    # 8
        
    ######################## H Layer 3  
    H_dense_3 = tf.concat([H_filter, conv1_H, conv2_H, conv2_MPF],axis=-1) # 101
    
    with tf.variable_scope('layer3_H'):
        weights  = tf.get_variable("w3",[3,3,101,31],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias     = tf.get_variable("b3",[31],initializer=tf.constant_initializer(0.0))
        conv3_H  = tf.nn.conv2d(H_dense_3, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_H  = tf.nn.relu(conv3_H)  # 31
        
        ################ MPF Layer 3     
    MPF_dense_3 = tf.concat([MPF_filter,conv1_MPF,conv2_MPF],axis=-1) # 28
    
    with tf.variable_scope('layer3_MPF'):
        weights   = tf.get_variable("w3",[3,3,28,8],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias      = tf.get_variable("b3",[8],initializer=tf.constant_initializer(0.0))
        conv3_MPF = tf.nn.conv2d(MPF_dense_3, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv3_MPF = tf.nn.relu(conv3_MPF)    # 8 
    
        
    ######################## H Layer 4  
    H_dense_4 = tf.concat([H_filter, conv1_H, conv2_H, conv3_H, conv3_MPF],axis=-1) # 31*5+24 159   132
    
    with tf.variable_scope('layer4_H'):
        weights  = tf.get_variable("w4",[3,3,132,68],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias     = tf.get_variable("b4",[68],initializer=tf.constant_initializer(0.0))
        conv4_H  = tf.nn.conv2d(H_dense_4, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_H  = tf.nn.relu(conv4_H) 
        
        weights  = tf.get_variable("w4_",[3,3,68,31],initializer=tf.truncated_normal_initializer(stddev=1e-3))      
        bias     = tf.get_variable("b4_",[31],initializer=tf.constant_initializer(0.0))
        conv4_H  = tf.nn.conv2d(conv4_H, weights, strides=[1,1,1,1], padding='SAME') + bias
        conv4_H  = tf.nn.relu(conv4_H) 
 
    mix_H_add = conv4_H   
    print('mix_H_add',mix_H_add.shape) 
    return mix_H_add
 








