#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:57:34 2018

@author: amax
"""

import numpy as np
import torch
import cv2
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, size, transform=None):
        fn = open(datatxt, 'r')
        data = []
        for line in fn:
            line = line.rstrip()
            words = line.split()
            data.append((words[0], int(words[1])))
            
        self.data = data
        self.transform = transform
        self.root = root
        self.size = size
        
    def __getitem__(self, index):
        img_name, label = self.data[index]
        img = cv2.imread(os.path.join(self.root,img_name))
        img = np.array(cv2.resize(img, (self.size,self.size))).transpose((2,0,1))
        img = (img-np.min(img))/(np.max(img)-np.min(img))-0.5    
        label = label-1
        return img, label
        
    def __len__(self):
        return len(self.data)



class GallantDataset(torch.utils.data.Dataset):
    def __init__(self, im_dir, datatxt, vs, size, transform=None):
        fn = open(datatxt, 'r')
        data = []
        for line in fn:
            line = line.rstrip()
            words = line.split()
            data.append((words[0], int(words[1])))
            
        self.data = data
        self.transform = transform
        self.im_dir = im_dir
        self.voxel = vs
        self.size = size
        
    def __getitem__(self, index):
        img_name, label = self.data[index]
        img = cv2.imread(os.path.join(self.im_dir,img_name))
        img = np.array(cv2.resize(img, (self.size,self.size))).transpose((2,0,1))
        img = (img-np.min(img))/(np.max(img)-np.min(img))-0.5   
        
        label = label-1
        v = self.voxel[index]
        img = np.expand_dims(img[0,:,:], 0)
        return img, label, v
        
    def __len__(self):
        return len(self.data)
    
    
if __name__ == '__main__':
    root_trn = '/home/amax/QK/Gallant_images/image_500/stimtrn/'
    datatxt_trn = '/home/amax/QK/Gallant_images/label/qk/stimtrn.txt'
    root_val = '/home/amax/QK/Gallant_images/image_500/stimval/'
    datatxt_val = '/home/amax/QK/Gallant_images/label/qk/stimval.txt'
    dataset_trn = MyDataset(root_trn, datatxt_trn, 224)
    dataset_val = MyDataset(root_val, datatxt_val, 224) 
    
    loader_trn = torch.utils.data.DataLoader(dataset_trn, 
                                             batch_size=16,
                                             shuffle=True,
                                             num_workers=2,
                                             drop_last=True)
    
    loader_val = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=2,
                                             drop_last=False)   

    area = 'LO'
    data_dir = '/home/amax/QK/Gallant_data'
    vs_trn = np.load(os.path.join(data_dir, '%strn_s1.npy'%(area)))
    vs_val = np.load(os.path.join(data_dir, '%sval_s1.npy'%(area)))
    vs_trn = np.asarray(vs_trn, dtype=np.float64)
    vs_val = np.asarray(vs_val, dtype=np.float64)
    
    gallantdataset_trn = GallantDataset(root_trn, datatxt_trn, vs_trn, 224)
    gallantdataset_val = GallantDataset(root_val, datatxt_val, vs_val, 224)
    
    gallantloader_trn = torch.utils.data.DataLoader(gallantdataset_trn,
                                                     batch_size=16,
                                                     shuffle=False,
                                                     num_workers=2,
                                                     drop_last=False)      
    gallantloader_val = torch.utils.data.DataLoader(gallantdataset_val,
                                                     batch_size=16,
                                                     shuffle=False,
                                                     num_workers=2,
                                                     drop_last=False)
    
    for i, data in enumerate(gallantloader_val):
        im, label, voxel = data
        plt.imshow(im[0].numpy().transpose((1,2,0))+0.5)       
        plt.show()
        print(label.shape)
        print(voxel.shape)



    
