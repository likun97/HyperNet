# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:42:57 2021

@author: LK
"""

import cv2
import numpy as np
import scipy.io as sio  
              
CAVE_data = sio.loadmat('./data/CAVE/CAVE20.mat')
train_ref = CAVE_data['ref']    
train_hsi = CAVE_data['hsi'] 
train_msi = CAVE_data['msi']
train_pan = CAVE_data['pan']
  



import tensorflow as tf
import skimage.measure
import os
  
nrtrain       = 198                                                         
EpochNum      = 251
batch_size    = 12                                                        
patch_size    = 8  
learning_rate = 0.001

tf.reset_default_graph()                                                        
X_output = tf.placeholder(tf.float32, shape=(batch_size, patch_size*32, patch_size*32,   31 ))   
H_input  = tf.placeholder(tf.float32, shape=(batch_size, patch_size,    patch_size,      31 ))     
M_input  = tf.placeholder(tf.float32, shape=(batch_size, patch_size*8,  patch_size*8,    3  ))             
P_input  = tf.placeholder(tf.float32, shape=(batch_size, patch_size*32, patch_size*32,   1  ))                             

 

from fusion_net_cave import compute_cost_l1_msssim, fusion_net_MAE3, fusion_net_MAE6           
PredX      = fusion_net_MAE3( M_input, P_input , H_input ) 
# PredX    = fusion_net_MAE6( M_input, P_input , H_input ) 


loss_l1, loss_msssim = compute_cost_l1_msssim(PredX , X_output)      
cost_all    = loss_l1 + 0.3*loss_msssim 



optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all) 
saver    = tf.train.Saver(tf.global_variables(), max_to_keep=100) 
tf.ConfigProto().gpu_options.allow_growth = True 

import time
start = time.time()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print('#'*60," Strart Training... ")

    # print to file      
    os.makedirs("./train_log/") 
    os.makedirs("./train_model/") 
    model_dir           = 'CAVE_MSSSIM03_MAE3_Epoch%d'%(EpochNum)  
    output_file_name    = "./train_log/epoch_%s.txt" %(model_dir)
    lk_output_file_name = "./train_log/batch_%s.txt" %(model_dir)

    for epoch_i in range(0, EpochNum):                                          
     
        Training_Loss = 0 
        randidx_all   = np.random.permutation(nrtrain)
                                                                                
        for batch_i in range(nrtrain // batch_size):                          
            randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

            batch_X = train_ref[randidx, :, :, :]     
            batch_M = train_msi[randidx, :, :, :]
            batch_P = train_pan[randidx, :, :, :]
            batch_H = train_hsi[randidx, :, :, :] 
            
            feed_dict = {M_input:batch_M,  P_input:batch_P,  H_input:batch_H,  X_output:batch_X }     
            cost_all_value,_  = sess.run([cost_all, optm_all] ,feed_dict = feed_dict ) 
            Training_Loss    += cost_all_value 

        # visualize output 
            _ ,ifshow = divmod(batch_i+1,300) 
            if ifshow ==1:  
                P_PredX  = sess.run(PredX , feed_dict = feed_dict )        
                P3       = P_PredX[0, :, :, :]
                                       
        # eval this batch       
                psnr     = skimage.metrics.peak_signal_noise_ratio(batch_X, P_PredX )        
                ssim     = skimage.metrics.structural_similarity  (batch_X, P_PredX, multichannel=True)
                nrmse    = skimage.metrics.normalized_root_mse    (batch_X, P_PredX )
                mse      = skimage.metrics.mean_squared_error     (batch_X, P_PredX)    
                CurLoss  = Training_Loss/(batch_i+1) 
                
                writing  = 'In %d epoch, Training_Loss =%.4f, PSNR =%.3f, SSIM =%.4f, NRMSE =%.4f\n' %(epoch_i+1, CurLoss,  psnr, ssim, nrmse)
                out_file = open(lk_output_file_name, 'a')
                out_file.write(writing)
                out_file.close()
        enter = open(lk_output_file_name, 'a')
        enter.write('\n')
        enter.close()
        
        output_data = "[%02d/%02d] loss_l1: %.4f, loss_msssim: %.4f \n" %(epoch_i+1, EpochNum, 
                                                          sess.run(loss_l1, feed_dict=feed_dict) , sess.run(loss_msssim, feed_dict=feed_dict))
        print(output_data)
        output_file = open(output_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
        
        # save model
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)
        if epoch_i <= 10:
            saver.save(sess, './train_model/%s/Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=True)
        else:
            if epoch_i % 5 == 0:
                saver.save(sess, './train_model/%s/Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
    end = time.time() 
    print("Train_Time", end - start)  
    sess.close()