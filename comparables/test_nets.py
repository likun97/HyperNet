# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:13:01 2021

@author: lk
"""  
import os
import time
import cv2
import numpy as np
import scipy.ndimage
import scipy.io as sio  

import torch                                
import torch.nn as nn
import torch.nn.functional as F  
from utils.metrics import ref_evaluate ,no_ref_evaluate  

from argparse import ArgumentParser
parser = ArgumentParser(description='Comparable-Nets')

parser.add_argument('--Network',       type=str,   default='Guided_DRCNN',  help='from {HyperPNN, Guided_DRCNN, DRCNN, GSA_SSR, Guided_DRCNN_ori, SSR}')
parser.add_argument('--Dataset',       type=str,   default='GF5_real',      help='training dataset from {CAVE, Harvard, Chikusei_v2, GF5_real, GF, Chikusei, CAVE_ori, GF_simu   }')
parser.add_argument('--testing_epoch', type=int,   default=550,             help='select a specific epoch number of trained models') 
parser.add_argument('--gpu_list',        type=str,   default='0',           help='gpu index') 
parser.add_argument('--trained_epoch',   type=int,   default=551,           help='epoch number of end training')
parser.add_argument('--model_dir',       type=str,   default='train_model', help='trained or pre-trained model directory') 
parser.add_argument('--fusion_test_dir', type=str,   default='fusion_test', help='fusion test of the pre-trained model')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 

import h5py 
ref,hsi,msi,pan  = [],[],[],[]
#################### Chikusei #######################  
if args.Dataset =='Chikusei' or args.Dataset =='Chikusei_v2':  
    ms_path  = 'E:\\0 HyperModel\\1_dealed_data\\Chikusei\\version_1\\test'   
    ms_file_list = os.listdir(ms_path)  
    ms_file_list.sort(key=lambda x:int(x.split('_test')[0]))                                
     
    for file in ms_file_list:                                                                 
        if not os.path.isdir(file):    
            mat_data = h5py.File(ms_path+"/"+file)     
            refs   = np.array(mat_data["I_ref"],dtype ='float32').T
            ref.append(refs)
            hsis   = np.array(mat_data["I_H"],dtype ='float32').T 
            hsi.append(hsis)   
            msis   = np.array(mat_data["I_M"],dtype ='float32').T  
            msi.append(msis)   
            pans   = np.array(mat_data["I_P"],dtype ='float32').T    
            pans   = np.expand_dims(pans,-1)          
            pan.append(pans)  
    print('ref.len' ,len(ref))   
#################### CAVE ############################ 
elif args.Dataset =='CAVE':  
    ms_path  = 'E:\\0 HyperModel\\1_dealed_data\\CAVE\\version_1\\test'                                                   
    ms_file_list = os.listdir(ms_path)  
    ms_file_list.sort(key=lambda x:int(x.split('_simu')[0]))                               
    for file in ms_file_list:                                                                 
        if not os.path.isdir(file):                                                             
            mat_data = h5py.File(ms_path+"/"+file)     
            refs   = np.array(mat_data["I_ref"],dtype ='float32').T
            ref.append(refs)
            hsis   = np.array(mat_data["I_H"],dtype ='float32').T 
            hsi.append(hsis)   
            msis   = np.array(mat_data["I_M"],dtype ='float32').T  
            msi.append(msis)   
            pans   = np.array(mat_data["I_P"],dtype ='float32').T    
            pans   = np.expand_dims(pans,-1)          
            pan.append(pans)  
    print('ref.len' ,len(ref))    
    #################### GF5_real #######################  
elif args.Dataset =='GF5_real':    
    ms_path  = 'E:\\0 HyperModel\\1_dealed_data\\GF5_real\\version_1\\test_front25'    

    ms_file_list = os.listdir(ms_path)  
    ms_file_list.sort(key=lambda x:int(x.split('_test')[0]))                        
     
    for file in ms_file_list:                                                                 
        if not os.path.isdir(file):    
            mat_data = h5py.File(ms_path+"/"+file)     
            refs   = np.array(mat_data["I_H"],dtype ='float32').T           
            #  ATTENTION #  No_Reference Experiment    \\test1_in_train  
            ref.append(refs)
            hsis   = np.array(mat_data["I_H"],dtype ='float32').T 
            hsi.append(hsis)   
            msis   = np.array(mat_data["I_M"],dtype ='float32').T  
            msi.append(msis)   
            pans   = np.array(mat_data["I_P"],dtype ='float32').T    
            pans   = np.expand_dims(pans,-1)          
            pan.append(pans)  
    print('ref.len' ,len(ref))   
    
else:
    raise NotImplementedError("Dataset [%s] is not recognized." % args.Dataset)

    

import sys
sys.path.append(".") 
from models.HyperPNN import HyperPNN
from models.GSA_SSR  import  GSA_SSRNET
from models.DRCNN import Guided_DRCNN, Guided_DRCNN_Test

if args.Network == "HyperPNN":
    Test_Network = HyperPNN
elif args.Network == "GSA_SSR":
    Test_Network = GSA_SSRNET
elif args.Network == "Guided_DRCNN":
    Test_Network = Guided_DRCNN_Test  
    # Guided_DRCNN, Guided_DRCNN_Test  
    # [ Guided_DRCNN_Test for real experiment ;Guided_DRCNN for all simulation expeeriment]
    
else:
    raise NotImplementedError("Network [%s] is not recognized." % args.Network) 

# =============================================================================
if args.Dataset =='CAVE' or args.Dataset =='CAVE_ori':
    in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands = 31,64,31,16,3,4,31      
elif args.Dataset =='Harvard':
    in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands = 31,64,31,15,3,3,31     
elif args.Dataset =='Chikusei_v2':
    in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands = 93,188,93,16,4,4,93
elif args.Dataset =='GF5_real':
    in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands = 180,362,180,15,4,5,180   
else:
    raise NotImplementedError("Network [%s] is not recognized." % args.Network) 
 
    
model = Test_Network(in_channels, mid_channels, out_channels, hp_ratio, n_select_bands, hm_ratio, n_bands)      



#%%
# Load pre-trained model with epoch number 
 
model       = nn.DataParallel(model)
model       = nn.DataParallel(model).to(device)  
model       = model.to(device) 
model_files = "Net_%s_Data_%s_Epoch_%d_V1" % (args.Network, args.Dataset, args.trained_epoch)    
model.load_state_dict(torch.load('./%s/%s/net_params_%d.pkl' % (args.model_dir, model_files, args.testing_epoch)))
    
print   ("Test_Network is:",model,'\n')
print   ("Model_File_Name is:\n",model_files,'\n')
 
with torch.no_grad():
    for img_id in range(len(ref)):      
        test_ref  =  ref[img_id]         
        test_hsi  =  hsi[img_id]
        test_msi  =  msi[img_id]      
        test_pan  =  pan[img_id]
        batch_R =  np.expand_dims(test_ref, 0)
        batch_H =  np.expand_dims(test_hsi, 0)
        batch_M =  np.expand_dims(test_msi, 0) 
        batch_P =  np.expand_dims(test_pan, 0) 
        batch_R = ((torch.tensor(batch_R).permute(0, 3, 1, 2)).type(torch.FloatTensor)).to(device) 
        batch_H = ((torch.tensor(batch_H).permute(0, 3, 1, 2)).type(torch.FloatTensor)).to(device)   
        batch_M = ((torch.tensor(batch_M).permute(0, 3, 1, 2)).type(torch.FloatTensor)).to(device) 
        batch_P = ((torch.tensor(batch_P).permute(0, 3, 1, 2)).type(torch.FloatTensor)).to(device) 
        
        start = time.time() 
        
        if args.Network == 'HyperPNN':
            batch_pred = model(batch_H, batch_M, batch_P) 
        elif args.Network == 'Guided_DRCNN':
            # _,_, batch_pred = model(batch_H, batch_M, batch_P, batch_R)    # Guided_DRCNN       --- Use it When testing Simulation data
            _, batch_pred = model(batch_H, batch_M, batch_P)                 # Guided_DRCNN_Test  --- Use it When testing GF_real    data-- No Reference Experiment Test
        elif args.Network == 'GSA_SSR':
            batch_pred, _, _,  _, _, _ = model(batch_H, batch_M, batch_P)     
        else:
            raise NotImplementedError("Network [%s] model error occurs." % args.Network)
            
        end  = time.time()
        tim  = end - start    
        batch_pred_np = batch_pred.permute(0, 2, 3, 1).cpu().data.numpy()  
        batch_pred_np = np.clip(batch_pred_np, 0, 1)    
        test_label    = batch_pred_np[0,:,:,:] 
        if (img_id==len(ref)-1):
            print('test_label',test_label.shape) 
    
        save_fusion_img_dir ="./%s/" %(args.fusion_test_dir) + model_files + "_Testepoch_%d/" %(args.testing_epoch)
        save_fusion_mat_dir ="./%s/" %(args.fusion_test_dir) + model_files + "_Testepoch_%d/" %(args.testing_epoch)
        save_fusion_log_dir ="./%s/" %(args.fusion_test_dir) + model_files + "_Testepoch_%d/" %(args.testing_epoch)
        
        if not os.path.exists(save_fusion_img_dir):
            os.makedirs(save_fusion_img_dir)
        if not os.path.exists(save_fusion_mat_dir):
            os.makedirs(save_fusion_mat_dir)
        if not os.path.exists(save_fusion_log_dir):
            os.makedirs(save_fusion_log_dir)
            
        if args.Dataset == 'Chikusei_v2' :
            cv2.imwrite (save_fusion_img_dir +'%d_test.png'%(img_id+1) ,np.uint8(255*test_label)[:, :, [10,20,30]] )   # CAVE        [5 ,15,25] 
            cv2.imwrite (save_fusion_img_dir +'%d_ref.png'%(img_id+1)  ,np.uint8(255*test_ref)  [:, :, [10,20,30]] )   # Chi    GF   [10,20,30] 
        elif args.Dataset == 'CAVE' or args.Dataset == 'Harvard': 
            cv2.imwrite (save_fusion_img_dir +'%d_test.png'%(img_id+1) ,np.uint8(255*test_label)[:, :, [5,15,25]] )    
            cv2.imwrite (save_fusion_img_dir +'%d_ref.png'%(img_id+1)  ,np.uint8(255*test_ref)  [:, :, [5,15,25]] )   
        elif args.Dataset == 'GF5_real':
            test_ref = scipy.ndimage.zoom(test_ref, (hp_ratio,hp_ratio,1), order=0)
            cv2.imwrite (save_fusion_img_dir +'%d_test.png'%(img_id+1) ,np.uint8(255*test_label)[:, :, [50,110,170]] )   #       
            cv2.imwrite (save_fusion_img_dir +'%d_exp.png'%(img_id+1)  ,np.uint8(255*test_ref)  [:, :, [50,110,170]] )  
               
        if args.Dataset == 'GF5_real':
            sio.savemat (save_fusion_mat_dir  +'Norm_%s_%d.mat'%(args.Network, img_id+1), { 'fusion':test_label} )
        else:
            sio.savemat (save_fusion_mat_dir  +'Norm_%s_%d.mat'%(args.Network, img_id+1), { 'ref':test_ref, 'fusion':test_label} )
        
           
        
        
        
        
        
        
        ref_results={}
        ref_results.update({'metrics: ':'  PSNR,   SSIM,   SAM,   ERGAS,  SCC,    Q,    RMSE'})
        no_ref_results={}
        no_ref_results.update({'metrics: ':'  D_lamda, D_s,    QNR'})
          
        
        temp_ref_results      = ref_evaluate( np.uint8(255*test_label), np.uint8(255*test_ref) )  
        # temp_no_ref_results = no_ref_evaluate( test_label,  LR_pan ,  LR_ms )    
     
        ref_results   .update({'xxx     ':temp_ref_results})
        # no_ref_results.update({'xxx     ':temp_no_ref_results})
        
         
      
        lk_output_file_ref    = save_fusion_log_dir+"%s_Dataset_ref.txt" % (args.Dataset)    
        # lk_output_file_no_ref = save_fusion_log_dir+"%s_Dataset_no_ref.txt" % (args.Dataset)     
        
       
        
        
        print('################## reference  #######################')
        for index, i in enumerate(ref_results):
            if index == 0:
                print(i, ref_results[i])
        else:    
                print(i, [round(j, 4) for j in ref_results[i]])
                
                
                list2str= str([ round(j, 4) for j in ref_results[i] ])
                list2str= ('%d  '+ list2str+'\n')%(img_id+1) 
                lk_output_file = open(lk_output_file_ref, 'a')
                lk_output_file.write(list2str)
                lk_output_file.close()  
        
        # print('################## no reference  ####################')
         
        # for index, i in enumerate(no_ref_results):
        #      if index == 0:9
        #          print(i, no_ref_results[i])
        #      else:    
        #          print(i, [round(j, 4) for j in no_ref_results[i]])
                
                
        #          list2str= str([ round(j, 4) for j in no_ref_results[i] ])     
        #          list2str=('%d  '+ list2str+'\n')%(img_id+1) 
        #          lk_output_file = open(lk_output_file_no_ref, 'a')
        #          lk_output_file.write(list2str)
        #          lk_output_file.close()  
        # print('#####################################################')
        
                 
        
      
    print('test finished') 

 




#%%



# =============================================================================
        # 根据上面GF的无参  要上采样 HSI 但也没用到 只是保证代码的完整性
        # if  args.Dataset=='GF':
        #        test_ref = scipy.ndimage.zoom(test_ref, (hp_ratio,hp_ratio,1), order=0)

               # test_ref  = tf.expand_dims(test_ref, 0)
               # test_ref  = tf.image.resize_images(images=test_ref, size=[ H_input.shape[1]*15 , H_input.shape[2]*15],method=tf.image.ResizeMethod.BICUBIC)
               # test_ref  = test_ref.eval(session=sess)
               # test_ref  = test_ref[0,:,:,:]
# =============================================================================
    



















