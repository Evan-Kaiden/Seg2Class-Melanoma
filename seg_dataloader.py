import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import os
import cv2
import pandas as pd

class segmentationLoader(Dataset):
    def __init__(self, root, image_paths, csv_path, image_size=(256,256)):
        self.root = root
        self.image_paths = image_paths
        self.csv = pd.read_csv(csv_path)
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.image_paths[idx])
        
        x = cv2.imread(image_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, self.image_size)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        seg_path = image_path.replace("Data", "GroundTruth").replace(base_name, base_name + "_Segmentation")
        seg_path = os.path.splitext(seg_path)[0] + ".png"

        seg = cv2.imread(seg_path)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        seg = cv2.resize(seg, self.image_size)

        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.')[0]
        label_row = self.csv[self.csv.image == image_name]
        label = label_row.label.values[0]

        if isinstance(label, str):
            if label == "benign":
                label = 0
            else:
                label = 1


        x = torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        seg = torch.tensor(seg.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.float32)

        return x, seg, label