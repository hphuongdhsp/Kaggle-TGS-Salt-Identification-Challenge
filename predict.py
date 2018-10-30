#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 07:06:54 2018

@author: ai
"""
import torch
import utils_pytorch
#import random
from loss_pytorch import LossBinary
import pandas as pd
from torch.autograd import Variable
import model
import Unes34_128
from new_loss import lovasz_hinge
from transforms import Scale,Centerpad,CenterCrop,DualCompose,VerticalFlip,Centerpad_V1
use_cuda = not False and torch.cuda.is_available()
from tqdm import tqdm
import numpy as np
checkpoint_path="/media/ai/C52E-4D0B/TGS/checkpoint_pytorch/"
path_scv='/media/ai/C52E-4D0B/TGS/scv/'

X_test,test_ids=utils_pytorch.load_test('/media/ai/C52E-4D0B/TGS/input')
def test_augment(img,mask=None,model='Centerpad'):
    if model=='scale':
        return Scale(size=224)(img,mask)
    else: 
        return Centerpad_V1(size=(224,224))(img, mask=None)
def test_augment_flip(img,mask=None,model='Centerpad'):
    if model=='scale':
        return Scale(size=224)(img,mask)
    else: 
        return DualCompose([VerticalFlip(prob=1),Centerpad_V1(size=(224,224))])(img, mask=None)
###############################################################################
device = torch.device("cuda" if use_cuda else "cpu")
MD0=Unes34_128.LinkNet34(num_classes=1, num_channels=3, pretrained=True).to(device)
criterion =LossBinary(jaccard_weight=0.3)
learning_rate = 1e-4  
optimizer = torch.optim.Adam(MD0.parameters(), lr=learning_rate)
utils_pytorch.load_checkpoint(checkpoint_path+"weights_LinkNet34_hyper_hinge_fold5.pth.tar", MD0, optimizer)
##############

##############

###############################################################################
salt_ID_dataset_test = utils_pytorch.saltIDDataset(X_test,masks=None,transform=test_augment)
salt_ID_dataset_test_flip = utils_pytorch.saltIDDataset(X_test,masks=None,transform=test_augment_flip)
img_size_ori = 101
img_size_target = 202

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
###############################################################################
    
all_predictions1 = []
for image in tqdm(torch.utils.data.DataLoader(salt_ID_dataset_test, batch_size = 30,drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)):
    image = image[0].type(torch.FloatTensor).cuda()
    y_pred = sigmoid(MD0(Variable(image)).cpu().data.numpy())
    all_predictions1.append(y_pred)

import cv2
all_predictions_stacked1 = np.vstack(all_predictions1)[:, 0, :, :]
del all_predictions1




preds_valid1 =all_predictions_stacked1.reshape(-1, 224, 224)
preds_valid1 = np.array([x[11:213,11:213] for x in preds_valid1])
preds_valid1 = np.array([downsample(x) for x in preds_valid1])
preds_valid=preds_valid1
del preds_valid1
###############################################################################
all_predictions1_flip = []
for image in tqdm(torch.utils.data.DataLoader(salt_ID_dataset_test_flip, batch_size = 30,drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)):
    image = image[0].type(torch.FloatTensor).cuda()
    y_pred = sigmoid(MD0(Variable(image)).cpu().data.numpy())
    all_predictions1_flip.append(y_pred)

all_predictions_stacked1_flip = np.vstack(all_predictions1_flip)[:, 0, :, :]
del all_predictions1_flip



preds_valid1_flip = all_predictions_stacked1_flip.reshape(-1, 224, 224)
preds_valid1_flip = np.array([x[11:213,11:213] for x in preds_valid1_flip])
preds_valid1_flip = np.array([downsample(x) for x in preds_valid1_flip])
preds_valid1_flip = np.array([vectical(x) for x in preds_valid1_flip])
preds_valid+=preds_valid1_flip

del preds_valid1_flip

device = torch.device("cuda" if use_cuda else "cpu")
MD1=model.LinkNet34(num_classes=1, num_channels=3, pretrained=True).to(device)
criterion =LossBinary(jaccard_weight=0.3)
learning_rate = 1e-4  
optimizer = torch.optim.Adam(MD1.parameters(), lr=learning_rate)
utils_pytorch.load_checkpoint(checkpoint_path+"weights_LinkNet34_fold1.pth.tar", MD1, optimizer)
###############################################################################
all_predictions2 = []
for image in tqdm(torch.utils.data.DataLoader(salt_ID_dataset_test, batch_size = 30,drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)):
    image = image[0].type(torch.FloatTensor).cuda()
    y_pred = sigmoid(MD1(Variable(image)).cpu().data.numpy())
    all_predictions2.append(y_pred)

all_predictions_stacked2 = np.vstack(all_predictions2)[:, 0, :, :]
del all_predictions2




preds_valid2 =all_predictions_stacked2.reshape(-1, 224, 224)
preds_valid2 = np.array([x[11:213,11:213] for x in preds_valid2])
preds_valid2 = np.array([downsample(x) for x in preds_valid2])
preds_valid+=preds_valid2
del preds_valid2
del all_predictions_stacked1
del all_predictions_stacked1_flip
###############################################################################
all_predictions2_flip = []
for image in tqdm(torch.utils.data.DataLoader(salt_ID_dataset_test_flip, batch_size = 30,drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)):
    image = image[0].type(torch.FloatTensor).cuda()
    y_pred = sigmoid(MD1(Variable(image)).cpu().data.numpy())
    all_predictions2_flip.append(y_pred)

all_predictions_stacked2_flip = np.vstack(all_predictions2_flip)[:, 0, :, :]
del all_predictions2_flip


preds_valid2_flip =all_predictions_stacked2_flip.reshape(-1, 224, 224)
preds_valid2_flip = np.array([x[11:213,11:213] for x in preds_valid2_flip])
preds_valid2_flip = np.array([downsample(x) for x in preds_valid2_flip])
preds_valid2_flip = np.array([vectical(x) for x in preds_valid2_flip])
preds_valid+=preds_valid2_flip
del preds_valid2_flip
del all_predictions_stacked2_flip
###############################################################################
MD2=model.LinkNet34(num_classes=1, num_channels=3, pretrained=True).to(device)
criterion = lovasz_hinge(per_image=False, ignore=None)
learning_rate = 1e-4  

#optimizer = torch.optim.SGD(MD2.parameters(), lr=5e-4, momentum=0.9)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, MD2.parameters()),
                          lr=0.0001, momentum=0.9, weight_decay=0.0001)
utils_pytorch.load_checkpoint(checkpoint_path+"weights_LinkNet34_fold2.pth.tar", MD2, optimizer)
###################################3
all_predictions3 = []
for image in tqdm(torch.utils.data.DataLoader(salt_ID_dataset_test, batch_size = 30,drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)):
    image = image[0].type(torch.FloatTensor).cuda()
    y_pred = sigmoid(MD2(Variable(image)).cpu().data.numpy())
    all_predictions3.append(y_pred)
all_predictions_stacked3 = np.vstack(all_predictions3)[:, 0, :, :]
del all_predictions3



preds_valid3 =all_predictions_stacked3.reshape(-1, 224, 224)
preds_valid3 = np.array([x[11:213,11:213] for x in preds_valid3])
preds_valid3 = np.array([downsample(x) for x in preds_valid3])
preds_valid+=preds_valid3
del preds_valid3
del all_predictions_stacked3
###############################################################################
all_predictions3_flip = []
for image in tqdm(torch.utils.data.DataLoader(salt_ID_dataset_test_flip, batch_size = 30,drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)):
    image = image[0].type(torch.FloatTensor).cuda()
    y_pred = sigmoid(MD2(Variable(image)).cpu().data.numpy())
    all_predictions3_flip.append(y_pred)

all_predictions_stacked3_flip = np.vstack(all_predictions3_flip)[:, 0, :, :]
del all_predictions3_flip


preds_valid3_flip =all_predictions_stacked3_flip.reshape(-1, 224, 224)
preds_valid3_flip = np.array([x[11:213,11:213] for x in preds_valid3_flip])
preds_valid3_flip = np.array([downsample(x) for x in preds_valid3_flip])
preds_valid3_flip = np.array([vectical(x) for x in preds_valid3_flip])
preds_valid+=preds_valid3_flip
del preds_valid3_flip
del all_predictions_stacked3_flip
preds_valid=preds_valid/6
np.save('/media/ai/C52E-4D0B/TGS/prob.npy',preds_valid)
###############################################################################

binary_prediction = (preds_valid > 0.505).astype(int)

test_file_list = [f.split('/')[-1].split('.')[0] for f in test_ids]
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

all_masks = []
for p_mask in list(binary_prediction):
    p_mask = rle_encoding(p_mask)
    all_masks.append(' '.join(map(str, p_mask)))
 
submit = pd.DataFrame([test_file_list,all_masks]).T
submit.columns = ['id','rle_mask']
submit.to_csv(path_scv+'ids.csv')
