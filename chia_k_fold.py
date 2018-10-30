#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 08:17:54 2018

@author: ai
"""

### file chia folder theo coverage. Cách sử dụng, sửa tên root chua file depths.csv va root chua anh, là file chứa ảnh train,
# trong file chứa ảnh train này gồm 2 thư mục images và masks 
## out put là 12 thư mục khác nhau, tương ứng 12 fold, gồm train và valid, 24 file csv ghi index của từng fold. 
import numpy as np
import pandas as pd
from sklearn.model_selection import  BaseCrossValidator
from tqdm import tqdm

import os
from keras.preprocessing.image import load_img


n_fold=6
root='/media/ai/C52E-4D0B/TGS/input/train/'
root_depth='/media/ai/C52E-4D0B/TGS/input/depths.csv'
#train_df = pd.read_csv(root+'train.csv', index_col='id', usecols=[0])
train_path_images = os.path.abspath(root +"images/")
train_ids = next(os.walk(train_path_images))[2]

depths = pd.read_csv(root_depth)
##load images and masks
D = [np.array(load_img(root+'images/'+idx, grayscale=True)) / 255 for idx in tqdm(train_ids)]
traindf=pd.DataFrame({'images':D},index=train_ids)
traindf["masks"] = [np.array(load_img(root+'masks/{}'.format(idx), grayscale=True)) / 255 for idx in tqdm(traindf.index)]
y=traindf.index 
traindf['index']=y
traindf['id']=y
for i in range(len(traindf['id'])):
    traindf['id'][i]=traindf['id'][i].replace('.png','')  
traindf.set_index('id', inplace=True)
#traindf=traindf.merge(depths, on='id', how='left')
#traindf.sort_values('z', inplace=True)
#traindf.drop('z', axis=1, inplace=True)
#traindf['fold'] = (list(range(n_fold))*depths.shape[0])[:traindf.shape[0]]
#traindf.drop(['masks','images'], axis=1, inplace=True)    
## load coverage: Counting the number of salt pixels in the masks and dividing them 
#by the image size. Also create 11 coverage classes, -0.1 having no salt at all to 1.0 being salt only.

traindf["coverage"] = traindf.masks.map(np.sum) / pow(101, 2)

def cov_to_class(val):    
    for i in range(0, 10):
        if val * 10 <= i :
            return i
n_fold=6        
traindf["coverage_class"] = traindf.coverage.map(cov_to_class)
traindf.sort_values('coverage_class', inplace=True)
traindf.drop(['masks','images'], axis=1, inplace=True)
### define number of fold
traindf['fold'] = (list(range(n_fold))*traindf.shape[0])[:traindf.shape[0]]
traindf.drop(['coverage','coverage_class'], axis=1, inplace=True)

traindf.to_csv(root+'fold.csv')
##

import shutil
class KFoldByTargetValue(BaseCrossValidator):
    def __init__(self, n_splits=1, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        sorted_idx_vals = sorted(zip(indices, X), key=lambda x: x[1])
        indices = [idx for idx, val in sorted_idx_vals]

        for split_start in range(self.n_splits):
            split_indeces = indices[split_start::self.n_splits]
            yield split_indeces

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_split
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)
cv = KFoldByTargetValue(n_splits=6, shuffle=True, random_state=1337)
for i, (train_index, valid_index) in enumerate(cv.split(traindf['fold'])):
    train_image=traindf['index'][train_index]
    valid_image=traindf['index'][valid_index]
    train={"index":train_image}
    valid={"index":valid_image}
    train_new=pd.DataFrame(train)
    valid_new=pd.DataFrame(valid)
    
    train_new.to_csv(root+"train_{}.csv".format(i))
    valid_new.to_csv(root+"valid_{}.csv".format(i))
    createFolder(root+'fold{}/train/images'.format(i))
    createFolder(root+'fold{}/train/masks'.format(i))
    createFolder(root+'fold{}/valid/images'.format(i))
    createFolder(root+'fold{}/valid/masks'.format(i))
    src_train=root+'images/'
    src_mask=root+'masks/'
    for file_name in train_image:
        full_file_name_image = os.path.join(src_train, file_name)
        if (os.path.isfile(full_file_name_image)):
            shutil.copy(full_file_name_image, root+'fold{}/train/images'.format(i)) 
        full_file_name_mask = os.path.join(src_mask, file_name)
        if (os.path.isfile(full_file_name_mask)):
            shutil.copy(full_file_name_mask, root+'fold{}/train/masks'.format(i))
    for file_name in valid_image:
        full_file_name_image = os.path.join(src_train, file_name)
        if (os.path.isfile(full_file_name_image)):
            shutil.copy(full_file_name_image, root+'fold{}/valid/images'.format(i)) 
        full_file_name_mask = os.path.join(src_mask, file_name)
        if (os.path.isfile(full_file_name_mask)):
            shutil.copy(full_file_name_mask, root+'fold{}/valid/masks'.format(i))
    
    





    
    
    