import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import Dataset

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        pretrained_res = resnet18(pretrained=True)
        self.features = nn.Sequential(pretrained_res.conv1,
                                      pretrained_res.bn1,
                                      pretrained_res.relu,
                                      pretrained_res.maxpool,
                                      pretrained_res.layer1,
                                      pretrained_res.layer2,
                                      pretrained_res.layer3,
                                      pretrained_res.layer4)

        self.classifiers = nn.Sequential(nn.Linear(512 ** 2, n_classes))

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        feature_size = x.size(2) * x.size(3)
        x = x.view(batch_size, 512, feature_size)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(
            batch_size, -1)
        x = torch.nn.functional.normalize(
            torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = self.classifiers(x)

        return x




