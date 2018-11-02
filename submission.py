#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 23:30:08 2018

@author: ai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 07:06:54 2018

@author: ai
"""
import torch
import utils
import pandas as pd
from torch.autograd import Variable
from lovazs_loss import lovasz_hinge
from transforms import (Scale,Centerpad,ImageOnly,Normalize,
                        DualCompose,HorizontalFlip)
use_cuda = not False and torch.cuda.is_available()
from tqdm import tqdm
import numpy as np
from Linknet34 import LinkNet34
checkpoint_path="/media/ai/C52E-4D0B/TGS/checkpoint_pytorch/"
path_scv='/media/ai/C52E-4D0B/TGS/scv/'


X_test,test_ids=utils.load_test('/media/ai/C52E-4D0B/TGS/input',101,101)
def test_augment(img,mask=None,model='scale'):
    if model=='scale':
        return DualCompose([Scale(size=128),
                            ImageOnly(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))])(img, mask=None)
    else: 
        return DualCompose([Centerpad(size=(128,128)),
                            ImageOnly(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))])(img, mask=None)
def test_augment_flip(img,mask=None,model='scale'):
    if model=='scale':
        return DualCompose([HorizontalFlip(prob=1),
                            Scale(size=128),
                            ImageOnly(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))])(img, mask=None)
    else: 
        return DualCompose([HorizontalFlip(prob=1),
                            Centerpad(size=(128,128)),
                            ImageOnly(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))])(img, mask=None)
###############################################################################
device = torch.device("cuda" if use_cuda else "cpu")
model=LinkNet34(num_classes=1, num_channels=3, pretrained=True).to(device)
criterion =lovasz_hinge(per_image=False, ignore=None) 
optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=0.01,
                    momentum=0.9,
                       weight_decay=0.0001) 
utils.load_checkpoint(checkpoint_path+"best_weights_third_step_LinkNet34_128_2.0.pth.tar", model, optimizer)

###############################################################################
salt_ID_dataset_test = utils.saltIDDataset(X_test,masks=None,transform=test_augment)
salt_ID_dataset_test_flip = utils.saltIDDataset(X_test,masks=None,transform=test_augment_flip)
img_size_ori = 101
img_size_target = 128

###############################################################################
    
all_predictions1 = []
for image in tqdm(torch.utils.data.DataLoader(salt_ID_dataset_test, batch_size = 30,drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)):
    image = image[0].type(torch.FloatTensor).cuda()
    y_pred = utils.sigmoid(model(Variable(image)).cpu().data.numpy())
    all_predictions1.append(y_pred)
all_predictions_stacked1 = np.vstack(all_predictions1)[:, 0, :, :]


preds_valid1 =all_predictions_stacked1.reshape(-1, 128, 128)
preds_valid1 = np.array([utils.downsample(x) for x in preds_valid1])
preds_valid=preds_valid1
del preds_valid1
###############################################################################
all_predictions1_flip = []
for image in tqdm(torch.utils.data.DataLoader(salt_ID_dataset_test_flip, batch_size = 30,drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True)):
    image = image[0].type(torch.FloatTensor).cuda()
    y_pred = utils.sigmoid(model(Variable(image)).cpu().data.numpy())
    all_predictions1_flip.append(y_pred)

all_predictions_stacked1_flip = np.vstack(all_predictions1_flip)[:, 0, :, :]
del all_predictions1_flip



preds_valid1_flip = all_predictions_stacked1_flip.reshape(-1, 128, 128)

preds_valid1_flip = np.array([utils.downsample(x) for x in preds_valid1_flip])
preds_valid1_flip = np.array([utils.horizontalFlip(x) for x in preds_valid1_flip])
preds_valid+=preds_valid1_flip
preds_valid=preds_valid/2
del preds_valid1_flip


###############################################################################
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_ids]
binary_prediction = (preds_valid > 0.5).astype(int)

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


submit = pd.DataFrame([test_file_list, all_masks]).T


submit.columns = ['id', 'rle_mask']
submit['rle_mask'].value_counts()
submit.to_csv(path_scv+'LinkNet34_size_128.csv', index = False)
