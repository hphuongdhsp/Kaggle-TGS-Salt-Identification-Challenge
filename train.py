"""
Created on Wed Sep  5 18:16:46 2018

@author: ai
"""
import numpy as np
import os
import utils

from learning_rate import CosineAnnealingLR_with_Restart
from tensorboardX import SummaryWriter
writer = SummaryWriter()
from loss_function import (get_iou_vector,
                    RobustFocalLoss2d)
from lovazs_loss import lovasz_hinge
import torch
import argparse
#from Unet34_pooling import LinkNet34
#from Resnet_ppd import LinkNet34
from Linknet34 import LinkNet34
#from resnet34_V4 import LinkNet34  ##from Unes34_128 import LinkNet34
from transforms import (DualCompose,ImageOnly,Normalize,HorizontalFlip,
                        OneOrOther,
                        VerticalFlip,OneOf,Scale,

                        ShiftScaleRotate,Distort2,Randompadding,
                        do_Gamma, RandomErasing,ShiftScale,Centerpad,Centerpad_V1,
                        Brightness_multiply,do_horizontal_shear,Brightness_shift,Randompadding_V1,Median_blur
                        )
"""
def train_transform(x, mask, prob=0.5):
    return DualCompose([HorizontalFlip()])(x, mask)
def val_transform(x,mask, prob=0.5):
    return Scale(128)(x, mask)
"""
    
parser = argparse.ArgumentParser(description='Salt Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--fold', type=int, default=2, metavar='N',
                        help='the fold we want to train (default: 1)')
parser.add_argument('--zeros-epochs', type=int, default=100, metavar='N',
                        help='number of first epochs to train (default: 10)')
parser.add_argument('--first-epochs', type=int, default=600, metavar='N',
                        help='number of first epochs to train (default: 10)')
parser.add_argument('--second-epochs', type=int, default=400, metavar='N',
                        help='number of second epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')

parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')

parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')

parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--model-name', type=str, default='LinkNet34_128', metavar='M',
                    help='model name (default: LinkNet34)')
parser.add_argument('--dataset-dir', type=str, default='/media/ai/C52E-4D0B/TGS', metavar='DD',
                    help='Where the dataset is saved to.')
parser.add_argument('--tensor_board', nargs='?', type=bool, default=True,
                    help='Show visualization(s) on tensor_board | True by  default')
parser.add_argument('--im-width', nargs='?', type=int, default=101,
                    help='Height of the input image')
parser.add_argument('--im-height', nargs='?', type=int, default=101,
                    help='Height of the input image')


args = parser.parse_args()  
if (args.im_width==101):
    pad = 128
else : 
    pad =256
def valid_augment(img,mask):
    return DualCompose([Scale(size=pad),ImageOnly(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))])(img, mask)

def train_augment(img,mask,prob=0.5):
    return DualCompose([HorizontalFlip(prob=0.5),ShiftScale(limit=4,prob=0.5),
                        OneOf([#ImageOnly(CLAHE(clipLimit=2.0, tileGridSize=(8, 8))),
                               ImageOnly(Brightness_shift(limit=0.1)),
                               ImageOnly(do_Gamma(limit=0.08)),                              
                               ImageOnly(Brightness_multiply(limit=0.08)),
                               ],prob=0.5),
                        ImageOnly(Median_blur( ksize=3, prob=.15)),
                        Scale(size=pad),
                        #Centerpad(size=(pad,pad)),
                        ImageOnly(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
                        #OneOf([Scale(size=128),Randompadding(size=(128,128))],prob=1),    
                        
                        #RandomErasing(probability = 0.22, sl = 0.02, sh = 0.2, r1 = 0.2, mean=[0.4914, 0.4822, 0.4465])
                        ])(img,mask)
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)      
createFolder(args.dataset_dir+'/checkpoint_pytorch/')
out_dir=args.dataset_dir+'/checkpoint_pytorch/sgdr_final'
createFolder(out_dir)

###############################################################################
###########                     LOAD DATA                        #######3######
X_train, Y_train=utils.load_train(args.dataset_dir+'/input/train/fold{}/train/'.format(args.fold),im_height=args.im_height,im_width=args.im_width)
X_valid, Y_valid=utils.load_train(args.dataset_dir+'/input/train/fold{}/valid/'.format(args.fold),im_height=args.im_height,im_width=args.im_width)

salt_ID_dataset_train = utils.saltIDDataset(X_train,Y_train, transform=train_augment)
salt_ID_dataset_val   = utils.saltIDDataset(X_train,Y_train,transform=valid_augment)

train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
                                           batch_size=args.batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_val, 
                                           batch_size=args.batch_size, 
                                           shuffle=False)

use_cuda = not False and torch.cuda.is_available()

torch.manual_seed(4200)

def save_checkpoint(state, is_best, filename = 'model_best.pth.tar'):
    if is_best:
        print ('Saved best!')
        torch.save(state, filename)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
###############################################################################
###                              train                                    #####
###############################################################################    
def do_train( model, device, train_loader, optimizer, criterion):
    model.train()
    #losses = []
    train_count=0
    train_loss =0
    for batch_idx, (data, target) in enumerate(train_loader):
        train_count += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() 
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        train_loss += loss     
       
        optimizer.step()

    train_loss/=train_count
    print('train loss: {:.5f}'.format(train_loss))
    return train_loss
###############################################################################
######                              valid                              3##3####
###############################################################################
def do_valid(model,criterion, valid_loader ):
    model.eval()
    valid_count=0
    valid_loss =0
    out = np.zeros(2,np.float32)
    score = []
    for inputs, truth in valid_loader:
        valid_count += 1
        inputs = inputs.cuda()
        truth = truth.cuda()
        with torch.no_grad():
            logit = model(inputs)
            loss  = criterion(logit, truth)
            
            score += [get_iou_vector(truth, (logit > 0).float()).item()]
            #score += [calc_metric(truth, (logit > 0).float()).item()]
        valid_loss += loss
    valid_loss/=valid_count
    valid_score = np.mean(score).astype(np.float32)
    out[0]=valid_loss
    out[1]=valid_score

    print('Valid loss: {:.5f}, score_1: {:.5f}'.format(valid_loss, valid_score))
    return out

#############################################################################33
device = torch.device("cuda" if use_cuda else "cpu")
model=LinkNet34(num_classes=1, num_channels=3, pretrained=True).to(device)


"""
setting different learning rate in the encoding and the decoding maybe improve the performance of model.

optimizer=torch.optim.SGD([{'params': model.decoder5.parameters()},
                            {'params': model.decoder4.parameters()},
                            {'params': model.decoder3.parameters()},
                            {'params': model.decoder2.parameters()},
                            {'params': model.decoder1.parameters()},
                            {'params': model.logit.parameters()},
                            {'params': model.hypercolumn.parameters()},
                            
                {'params': model.conv1.parameters(),   'lr': 0.01},
                {'params': model.maxpool.parameters(), 'lr': 0.01},
                {'params': model.encoder1.parameters(),'lr': 0.01},
                {'params': model.encoder2.parameters(),'lr': 0.01},
                {'params': model.encoder3.parameters(),'lr': 0.01},
                {'params': model.encoder4.parameters(),'lr': 0.01},
                {'params': model.center.parameters(),  'lr': 0.01}
            ], lr=0.025, momentum=0.9,weight_decay=0.0001)
"""
print('---------first step --------')
print('train 60 first epochs with RobustFocalLoss2d loss')
criterion =RobustFocalLoss2d(gamma=2, size_average=True)
optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=0.01,
                    momentum=0.9,
                       weight_decay=0.0001) 
best_score=0.5
for epoch in range(1,60):
        
    print('epoch : ', epoch) 
    lr2 = optimizer.param_groups[0]['lr']
    print('decoding_lr:',lr2)                    
    do_train( model, device, train_loader, optimizer, criterion)        
    valid_loss=do_valid(model, criterion, val_loader)
    is_best=valid_loss[1]>best_score
    best_score=max(best_score,valid_loss[1])
    utils.save_checkpoint(args.dataset_dir+'/checkpoint_pytorch/best_weights_first_step_{}_{:.1f}.pth.tar'.format(args.model_name, args.fold), model, optimizer)
    
print('---------second stade--------')
print('train next 150 epoch with lovasz loss')

utils.load_checkpoint(args.dataset_dir+'/checkpoint_pytorch/best_weights_first_step_{}_{:.1f}.pth.tar'.format(args.model_name, args.fold), model,optimizer)
criterion =lovasz_hinge(per_image=True, ignore=None)
optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=0.001,
                    momentum=0.9,
                       weight_decay=0.0001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                               factor=0.5, patience=8, 
                                               min_lr=1e-5)

for epoch in range(1,100):        
    print('epoch : ', epoch) 
    lr2 = optimizer.param_groups[0]['lr']
    print('decoding_lr:',lr2)                    
    do_train( model, device, train_loader, optimizer, criterion)        
    valid_loss=do_valid(model, criterion, val_loader)    
    scheduler.step(valid_loss[0])    
    is_best=valid_loss[1]>best_score
    best_score=max(best_score,valid_loss[1])
    utils.save_checkpoint(args.dataset_dir+'/checkpoint_pytorch/best_weights_second_step_{}_{:.1f}.pth.tar'.format(args.model_name, args.fold), model, optimizer)
    
utils.load_checkpoint(args.dataset_dir+'/checkpoint_pytorch/best_weights_second_step_{}_{:.1f}.pth.tar'.format(args.model_name, args.fold), model,optimizer)
###############################################################################
##################                 third step             #####################

print('---------third stade--------')
print('---------snap_shot_stade--------')
optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=0.0004,
                    momentum=0.9,
                       weight_decay=0.0001)

scheduler=CosineAnnealingLR_with_Restart(optimizer, T_max=50, 
                                             T_mult=1, model=model, 
                                             out_dir=out_dir, 
                                             take_snapshot=True, eta_min=0.0001, last_epoch=-1)
for epoch in range(1,151):
    scheduler.step()        
    print('epoch : ', epoch) 
    lr2 = optimizer.param_groups[0]['lr']
    print('decoding_lr:',lr2)                    
    do_train( model, device, train_loader, optimizer, criterion)        
    valid_loss=do_valid(model, criterion, val_loader)
    is_best=valid_loss[1]>best_score
    best_score=max(best_score,valid_loss[1])
    utils.save_checkpoint(args.dataset_dir+'/checkpoint_pytorch/best_weights_third_step_{}_{:.1f}.pth.tar'.format(args.model_name, args.fold), model, optimizer)
utils.save_checkpoint(args.dataset_dir+'/checkpoint_pytorch/best_weights_third_step_{}_{:.1f}.pth.tar'.format(args.model_name, args.fold), model, optimizer)
