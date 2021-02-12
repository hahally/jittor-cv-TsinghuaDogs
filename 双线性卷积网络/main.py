from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch

from base_model import Net
from load_data import LoadData

import warnings
import json
import os
import time
import copy
from tqdm import tqdm
import numpy as np
from math import sqrt
import cv2
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def evaluate_accuracy(data_iter, net):

    acc_sum, n = 0.0, 0
    for X, y in tqdm(data_iter):
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train(net, criterion, optimizer, loss, device, dataloaders, epochs=1, path='./'):

    batch_count = 0
    best_acc = 0
    for epoch in range(epochs):
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()

        for x, y in tqdm(dataloaders['train']):
            x, y = x.to(device), y.to(device)

            out = net(x)

            train_loss = loss(out, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_sum += train_loss.item()
            train_acc_sum += (out.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
            # break
        val_acc = evaluate_accuracy(dataloaders['valid'], net)
        print('epoch % d, loss % .4f, train acc % .3f, test acc % .3f, time % .1f sec'% (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, val_acc, time.time() - start))
        if best_acc<val_acc:
            best_acc = val_acc
            time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
            torch.save(net.state_dict(), os.path.join(path,'{}-best_model.pth'.format(time_stamp)))
            print('Saved model: ',time_stamp)
        # break
def main():
    # ------------------ config ------------------ #
    n_classes = 130
    batch_size = 10
    epochs = 25
    save_model_path = './'

    # -------------------------------------------- #

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset = LoadData(data_dir='./dataset/JsonData/',
                          mode='train',
                          transforms=data_transforms['train'])

    train_loaders = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    dataset = LoadData(data_dir='./dataset/JsonData/',
                          mode='valid',
                          transforms=data_transforms['valid'])

    valid_loaders = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    dataloaders = {
        'train': train_loaders,
        'valid': valid_loaders
    }

    print('GPUs Available:', torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on ", device)
    
    net = Net(n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()

    train(net, criterion, optimizer, loss, device, dataloaders,epochs ,save_model_path)

if __name__=='__main__':
    main()