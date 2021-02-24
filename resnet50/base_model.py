import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision.models import resnet18,resnet50,vgg16,densenet121
from torchvision import transforms
from torch.utils.data import Dataset

class Net(nn.Module):
    def __init__(self, n_classes,pretrained=False):
        super(Net, self).__init__()
        self.pretrained_res = resnet50(pretrained=pretrained)
        if pretrained == True:
            for param in self.pretrained_res.parameters():
                param.requires_grad = False

        self.pretrained_res.fc = nn.Linear(2048, n_classes)
        
        

    def forward(self, x):

        return self.pretrained_res(x)