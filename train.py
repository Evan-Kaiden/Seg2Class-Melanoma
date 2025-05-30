import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seg_dataloader import segmentationLoader



train_images = "/Users/evankaiden/Documents/Seg2Class-Melanoma/isic 2016/ISBI2016_ISIC_Part1_Training_Data"
train_csv = "/Users/evankaiden/Documents/Seg2Class-Melanoma/isic 2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"

test_images = "/Users/evankaiden/Documents/Seg2Class-Melanoma/isic 2016/ISBI2016_ISIC_Part1_Test_Data"
test_csv = "/Users/evankaiden/Documents/Seg2Class-Melanoma/isic 2016/ISBI2016_ISIC_Part3B_Test_GroundTruth.csv"


train_dataset = segmentationLoader(train_images, os.listdir(train_images), train_csv)
test_dataset = segmentationLoader(test_images, os.listdir(test_images), test_csv)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader =  DataLoader(test_dataset, batch_size=16, shuffle=True)


for x,y,z in test_loader:
    print(x.shape)
    print(y.shape)
    print(z.shape)
    break