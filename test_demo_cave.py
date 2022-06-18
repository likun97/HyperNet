# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:15:03 2021

@author: DELL
"""

import os
import h5py
import numpy as np
import scipy.io as sio
 


ms_path  = './data/CAVE/test'  

ms_file_list = os.listdir(ms_path)  
ms_file_list.sort(key=lambda x:int(x.split('_')[0]))                                           
ref,hsi,msi,pan  = [],[],[],[] 

for file in ms_file_list:                                                                 
    if not os.path.isdir(file):     
    #################### CAVEdata ####################### 
    #################### CAVEdata #######################                                                 
        mat_data = sio.loadmat(ms_path+"/"+file)                                           
        
        refs   = mat_data['I_ref']   
        ref.append(refs)
        hsis   = mat_data['I_H']   
        hsi.append(hsis)        
        msis   = mat_data['I_M']   
        msi.append(msis)    
        pans   = mat_data['I_P']  
        pans   = np.expand_dims(pans,-1)          
        pan.append(pans)  
        
print('ref.len' ,len(ref)) 



import os
import cv2
import numpy as np 
import tensorflow as tf
from metrics import ref_evaluate #,no_ref_evaluate  
 # tf.reset_default_graph() 
   
## CAVEdata ##
patch_size =16                                                        
X_output   = tf.placeholder(tf.float32, shape=(1, patch_size*32, patch_size*32,   31 ))   
H_input    = tf.placeholder(tf.float32, shape=(1, patch_size,    patch_size,      31 ))     
M_input    = tf.placeholder(tf.float32, shape=(1, patch_size*8,  patch_size*8,    3  ))             
P_input    = tf.placeholder(tf.float32, shape=(1, patch_size*32, patch_size*32,   1  ))    
 
 
from fusion_net_cave import fusion_net_MAE3  

PredX          = fusion_net_MAE3( M_input, P_input , H_input )  
model_path     = './train_model/CAVE_MSSSIM03_MAE3_Epoch251/'      
save_path_save =               'CAVE_MSSSIM03_MAE3_Epoch251/'       # 
 


saver  = tf.train.Saver(max_to_keep = 5)
config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True) 
config.gpu_options.allow_growth = True
 
import time
time_all =[]
with tf.Session(config=config) as sess:  
      
   ckpt = tf.train.latest_checkpoint(model_path)
   saver.restore(sess, ckpt) 
    
   for num in range(len(ref)):      
       test_ref  =  ref[num]        
       test_hsi  =  hsi[num]
       test_msi  =  msi[num]      
       test_pan  =  pan[num]
       
       batch_M =  np.expand_dims(test_msi, 0) 
       batch_P =  np.expand_dims(test_pan, 0)
       batch_H =  np.expand_dims(test_hsi, 0) 
       time_start = time.time()   
       one = sess.run(PredX , feed_dict={M_input:batch_M,  P_input:batch_P,  H_input:batch_H } ) 
       time_end = time.time()    
       time_con = time_end - time_start    
       time_all.append(time_con)
  
       one = np.clip(one, 0, 1)  
       test_label = one[0,:,:,:] 
       if (num==len(ref)-1):
           print('test_label',test_label.shape)  
       save_testimage_dir='./test_reduced/' +save_path_save
       save_test_mat_dir ='./test_reduced_mat/' +save_path_save 
       if not os.path.exists(save_testimage_dir):
           os.makedirs(save_testimage_dir)
       if not os.path.exists(save_test_mat_dir):
           os.makedirs(save_test_mat_dir)
     
       cv2.imwrite (save_testimage_dir +'%d_test.png'%(num+1) ,np.uint8(255*test_label)[:, :, [5,15,25]] ) 
       cv2.imwrite (save_testimage_dir +'%d_ref.png'%(num+1)  ,np.uint8(255*test_ref)  [:, :, [5,15,25]] ) 
       # save as uint8
       sio.savemat (save_test_mat_dir  +'Hyper_%d.mat'%(num+1), { 'ref':np.uint8(255*test_ref), 'fusion':np.uint8(255*test_label)} )
       
       gt = test_ref 
       ref_results={}
       ref_results.update({'metrics: ':'  PSNR,   SSIM,   SAM,   ERGAS,  SCC,    Q,    RMSE'})
       no_ref_results={}
       no_ref_results.update({'metrics: ':'  D_lamda, D_s,    QNR'})
         
       temp_ref_results      = ref_evaluate( np.uint8(255*test_label), np.uint8(255*test_ref) ) 
       # temp_no_ref_results = no_ref_evaluate( test_label,  LR_pan ,  LR_ms )    
       ref_results   .update({'xxx     ':temp_ref_results})
       # no_ref_results.update({'xxx     ':temp_no_ref_results})
       
       save_testlog_dir='./test_reduced_log/' + save_path_save 
       if not os.path.exists(save_testlog_dir):
           os.makedirs(save_testlog_dir) 
       lk_output_file_ref    = save_testlog_dir+"log.txt"    
       
       print('################## reference  #######################')
       for index, i in enumerate(ref_results):
           if index == 0:
               print(i, ref_results[i])
       else:    
               print(i, [round(j, 4) for j in ref_results[i]])  
               list2str= str([ round(j, 4) for j in ref_results[i] ])
               list2str= ('%d  '+ list2str+'\n')%(num+1) 
               lk_output_file = open(lk_output_file_ref, 'a')
               lk_output_file.write(list2str)
               lk_output_file.close()  
       
       # print('################## no reference  ####################') 
       # for index, i in enumerate(no_ref_results):
       #      if index == 0:
       #          print(i, no_ref_results[i])
       #      else:    
       #          print(i, [round(j, 4) for j in no_ref_results[i]]) 
       #          list2str= str([ round(j, 4) for j in no_ref_results[i] ])     
       #          list2str=('%d  '+ list2str+'\n')%(num+1) 
       #          lk_output_file = open(lk_output_file_no_ref, 'a')
       #          lk_output_file.write(list2str)
       #          lk_output_file.close()  
       # print('#####################################################')
        
     
   print('test finished') 