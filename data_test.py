import os
from os import listdir
import json
import numpy as np
import torch

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DataTest(Dataset):
    def __init__(self, args, mode):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir)

        print(self.img_dir)

        ''' read the data list '''
        list = []
        for f in sorted(listdir(self.img_dir)):
            img_path = os.path.join(self.data_dir, f)
            list.append(img_path)

        array = np.array(list)
        self.data = array

        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                transforms.Normalize(MEAN, STD)
            ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                transforms.Normalize(MEAN, STD)
            ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' get data '''
        img_path = self.data[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img)
