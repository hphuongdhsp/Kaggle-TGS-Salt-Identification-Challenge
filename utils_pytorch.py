#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:01:17 2018

@author: ai
"""
import numpy as np
from tqdm import tqdm
import os
import cv2
import torch
import torch.utils.data
#from skimage.transform import resize
#import loss
#import models
img_size_ori=101
im_width = 101
im_height = 101
im_chan = 3
img_size_target=128
def load_train(path,im_height,im_width):
    train_path_images = os.path.abspath(path +"/images/")
    train_ids = next(os.walk(train_path_images))[2]
    X_train = np.zeros((len(train_ids), im_height, im_width,im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), im_height,im_width), dtype=np.uint8)
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        img = cv2.imread(path + '/images/' + id_,1)
        img = cv2.resize(img, (im_height, im_width), interpolation=cv2.INTER_LINEAR)
        X_train[n] = img
        mask = cv2.imread(path + '/masks/' + id_,0)
        mask = cv2.resize(mask, (im_height, im_width), interpolation=cv2.INTER_LINEAR)
        Y_train[n] = (mask>0).astype(np.uint8)#(resize(mask, (128, 128, 1),
                     #   mode='constant', 
                     #   preserve_range=True)>0).astype(np.uint8)
    return X_train, Y_train


def load_test(path,im_height,im_width):
    test_path_images = os.path.abspath(path + "/test/images/")
    test_ids = next(os.walk(test_path_images))[2]
    X_test = np.zeros((len(test_ids), im_height, im_width,im_chan), dtype=np.uint8)
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = cv2.imread(path + '/test/images/' + id_,1)
        img = cv2.resize(img, (im_height,im_width),interpolation=cv2.INTER_LINEAR)
        X_test[n] = img
    return X_test, test_ids


class saltIDDataset(torch.utils.data.Dataset):

    def __init__(self,images, masks=None,transform=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.images=images
        self.transform = transform
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.masks is None:
            image,_=self.transform(self.images[idx])
            image=image
            return (to_float_tensor(image), )
        else:    
            image = self.images[idx]
            mask = self.masks[idx]
            image,mask = self.transform(image,mask)
            image=image
           
            return to_float_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
        
def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return cv2.resize(img, (img_size_ori, img_size_ori))
def vectical(img):    
    return cv2.flip(img, 0)

def centercrop(mask):# hxw
    h,w,=mask.shape
    h_crop=int((h-101)/2)
    w_crop=int((w-101)/2)
    mask=mask[h_crop:h_crop+101,w_crop:w_crop+101]
    return mask

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
