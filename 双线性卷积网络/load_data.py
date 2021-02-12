import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")


class LoadData(Dataset):
    def __init__(self,data_dir,mode='train',transforms=None):
        super().__init__()
        self.mode = mode.upper()
        self.data_dir = data_dir
        self.transforms = transforms

        assert self.mode in ['TRAIN', 'VALID']

        with open(os.path.join(self.data_dir, self.mode + '_images.json'), 'r') as j:
            self.images = json.load(j)

        # self.imgs = []

        # for item in self.images:
        #     label = int(item.split('-')[1][-3:]) - 1
        #     if label in [0,1,2,3,4,5,6,7,8,9]:
        #         self.imgs.append(item)

        self.total_len = len(self.images)

        print("[*] Loading {} {} images.".format(self.mode, self.total_len))

    def __getitem__(self, idx):
        
        img = Image.open('./dataset/low-resolution/'+self.images[idx]).convert('RGB')
        label = int(self.images[idx].split('-')[1][-3:]) - 1
        if self.transforms is not None:
            img = self.transforms(img)
    
        return img, label
    
    def __len__(self):
        return len(self.images)

# ----------------------------------------------------------------------- #
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'valid': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
# t = LoadData(data_dir='./dataset/JsonData/',transforms=data_transforms['train'])

# dataloaders = torch.utils.data.DataLoader(t, batch_size=2, shuffle=True)

# for x,y in dataloaders:
#     print(x.shape,y.shape)