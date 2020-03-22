import os
from os import listdir
import json
import numpy as np
import torch
import random

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DataC(Dataset):
    def __init__(self, args, mode):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, self.mode + '/img/')
        self.label_dir = os.path.join(self.data_dir, self.mode + '/seg/')

        ''' read the data list '''
        list = []
        for f in sorted(listdir(self.img_dir)):
            list.append(f)
            list.append(f)

        array = np.array(list)
        array = array.reshape(-1,2)
        self.data = array

        help_list = []

        ''' set up image path '''
        for d in self.data:
            im_path = os.path.join(self.img_dir, d[0])
            lb_path = os.path.join(self.label_dir, d[1])
            help_list.append(im_path)
            help_list.append(lb_path)

        help_array = np.array(help_list)
        help_array = help_array.reshape(-1, 2)
        self.data = help_array


        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                #transforms.RandomHorizontalFlip(0.5),
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
        img_path, lbl_path = self.data[idx]


        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        cls = Image.open(lbl_path).convert('L')

        if random.random() > 0.6:
            angle = random.randint(-30, 30)
            img.rotate(angle)
            cls.rotate(angle)

        cls = np.array(cls)
        cls = torch.from_numpy(cls)
        cls = cls.long()

        return self.transform(img), cls


