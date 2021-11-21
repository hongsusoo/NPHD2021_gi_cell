import os
import pandas as pd
from PIL import Image

import wandb
import math

from ipywidgets import IntProgress
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from time import time

import torch
import torch.utils.data as data

import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
# import albumentations.pytorch

import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
import timm
import sklearn.metrics

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model', type=str, default='efficientnet-b0')
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--kfold_train', type=str, default=None)
parser.add_argument('--kfold_val', type=str, default=None)
# parser.add_argument('--classnum', type=int, default=2)
parser.add_argument('--save_path', type=str, default='data/val')

args = parser.parse_args()

device = torch.device("cuda:0")
batch_size = 16
class_n = 2
learning_rate = args.lr
epochs = args.epoch
save_path = args.save_path

wandb.init(project='NPDH', name=args.model + args.kfold_train + str(learning_rate), ##arg
    config={
    "batch size": batch_size,
    "epochs" : epochs,
    "learning rate": learning_rate,
})

# Set random seed
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) 
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = True

train = pd.read_csv(args.kfold_train) ##arg
val = pd.read_csv(args.kfold_val)

class CustomDataset(Dataset):
    def __init__(self, files, labels=None, mode='train', transform = None):
        self.mode = mode
        self.files = files
        self.transform = transform
        if mode == 'train':
            self.labels = labels
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        if self.mode == 'train':
            img = cv2.imread(self.files[i])
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)/255
            # img = np.transpose(img, (2,0,1))
            if self.transform is not None:
                transformed = self.transform(image=img)
                img = transformed["image"]
                
            return {
                'img' : img,
                'label' : torch.tensor(self.labels[i], dtype=torch.long)
            }
        else:
            img = cv2.imread(self.files[i])
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)/255
            if self.transform is not None:
                transformed = self.transform(image=img)
                img = transformed["image"]
            # img = np.transpose(img, (2,0,1))
            return {
                'img' : img
            }

train_transform = A.Compose([
                    A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=1, always_apply=False, p=0.05),
                    A.Rotate(always_apply=False, p=1.0, limit=(-90, 90), interpolation=0, border_mode=0, value=(255, 255, 255), mask_value=None),
                    A.RandomResizedCrop(always_apply=False, p=0.1, height=512, width=512, scale=(0.5, 1.0), ratio=(0.75, 1.3333333730697632), interpolation=0),
                    A.HorizontalFlip(p=0.1),
                    A.VerticalFlip(p=0.1),
                    A.HueSaturationValue(always_apply=False, p=0.2, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20)),
                    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=4, p=0.1),
                    A.ElasticTransform(p=0.2),
                    ToTensorV2()
                          ])
val_transform = A.Compose([
                          ToTensorV2()
                          ])



train_dataset = CustomDataset(train['image_path'], train['labels'].values, transform = train_transform)
val_dataset = CustomDataset(val['image_path'], val['labels'].values, transform = val_transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)


from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained(args.model,num_classes=class_n)


model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

def evaluation(ground_truth, pred_output, labels = [1,0]):
    mtris = sklearn.metrics.confusion_matrix(ground_truth,pred_output, labels=labels)
    TP, FN = mtris[0]
    FP, TN = mtris[1]
    ttl = TP+FN+FP+TN
    accuracy = (TP+TN)/ttl
    specificity = TN/(TN+FP)
    sensitivity = TP/(TP+FN)
    precision = TP/(TP+FP)
    negative_predicable_value = TN/(TN+FN)
    F1score = 2*precision*sensitivity/(precision+sensitivity)

    return accuracy,specificity,sensitivity,precision,negative_predicable_value,F1score

def train_step(batch_item, epoch, batch, training):
    img = batch_item['img'].to(device)
    label = batch_item['label'].to(device)
    ground_truth = []
    pred_output = []
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img)
            loss = criterion(output, label)
            pred_output+=output.argmax(axis =1).cpu().tolist()
            ground_truth+=label.cpu().tolist()
        loss.backward()
        optimizer.step()
        return loss,ground_truth,pred_output
    else:
        model.eval()
        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)
            pred_output+=output.argmax(axis =1).cpu().tolist()
            ground_truth+=label.cpu().tolist()
        return loss,ground_truth,pred_output

loss_plot, val_loss_plot = [], []

for epoch in range(epochs):
    print('epoch :', epoch)
    total_loss, total_val_loss = 0, 0
    
    tqdm_dataset = tqdm(enumerate(train_dataloader))
    training = True
    ground_truth = []
    pred_output = []   
    for batch, batch_item in tqdm_dataset:
        batch_loss,gt,po = train_step(batch_item, epoch, batch, training)
        total_loss += batch_loss
        ground_truth+=gt
        pred_output+=po
        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Loss': '{:06f}'.format(batch_loss.item()),
            'Total Loss' : '{:06f}'.format(total_loss/(batch+1))
        })
    loss_plot.append(total_loss/(batch+1))
    acc,spec,sens,prec,npv,f1 = evaluation(ground_truth,pred_output)

    wandb.log({
            'train Total Loss' : '{:06f}'.format(total_loss/(batch+1)),
            "train Accuracy" : acc, 
            "train Specificity" : spec,
            "train Sensitivity" : sens,
            "train Precision" : prec,
            "train Negative_Predicable_Value" : npv,
            "train F1score" : f1,
            "train total_mean" : (acc+spec+sens+prec+npv+f1)/6
        })

    tqdm_dataset = tqdm(enumerate(val_dataloader))
    training = False
    ground_truth = []
    pred_output = []   
    for batch, batch_item in tqdm_dataset:
        batch_loss,gt,po = train_step(batch_item, epoch, batch, training)
        total_val_loss += batch_loss
        ground_truth+=gt
        pred_output+=po       
        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Val Loss': '{:06f}'.format(batch_loss.item()),
            'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1))
        })
    acc,spec,sens,prec,npv,f1 = evaluation(ground_truth,pred_output)
    wandb.log({
            "val Total Loss": '{:06f}'.format(total_val_loss/(batch+1)),
            "val Accuracy" : acc, 
            "val Specificity" : spec,
            "val Sensitivity" : sens,
            "val Precision" : prec,
            "val Negative_Predicable_Value" : npv,
            "val F1score" : f1,
            "val total_mean" : (acc+spec+sens+prec+npv+f1)/6
    })
    ttt= (acc+spec+sens+prec+npv+f1)/6
    val_loss_plot.append(total_val_loss/(batch+1))
    
    torch.save(model, f'{save_path}_{epoch+1}_{ttt:0.4f}.pth')