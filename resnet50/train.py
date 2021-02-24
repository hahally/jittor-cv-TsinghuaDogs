import time
import os
from tqdm import tqdm

import torch


import torch.nn.functional as F
def add_loss(out1,out2):
    D_ec = F.pairwise_distance(out1, out2, p=2)
    
    return D_ec

def evaluate_accuracy(data_iter, net, device):

    acc_sum, n = 0.0, 0
    net.eval()
    for x, y in tqdm(data_iter):
        
        x1 = x[0:int(x.size(0)/2)]
        x2 = x[int(x.size(0)/2):]
        y1 = y[0:int(y.size(0)/2)]
        y2 = y[int(y.size(0)/2):]
        x1, y1 = x1.to(device), y1.to(device)
        x2, y2 = x2.to(device), y2.to(device)
        
        out1 = net(x1)
        out2 = net(x2)

        acc_sum += (out1.argmax(dim=1) == y1).float().sum().item()
        acc_sum += (out2.argmax(dim=1) == y2).float().sum().item()
        
        n += y.shape[0]
        
    return acc_sum / n

def train_model(net, optimizer, loss, scheduler,device, dataloaders, epochs=1, path='./'):

    best_acc = 0
    for epoch in range(epochs):
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        scheduler.step()
        net.train()
        for x, y in tqdm(dataloaders['train']):
            assert x.size(0)%2 == 0

            x1 = x[0:int(x.size(0)/2)]
            x2 = x[int(x.size(0)/2):]
            y1 = y[0:int(y.size(0)/2)]
            y2 = y[int(y.size(0)/2):]
            x1, y1 = x1.to(device), y1.to(device)
            x2, y2 = x2.to(device), y2.to(device)

            gamma = (y1==y2) + 0
            lam = 0.5
            
            out1 = net(x1)
            out2 = net(x2)
            
            train_loss1 = loss(out1, y1)
            train_loss2 = loss(out2, y2)

            D_ec = add_loss(out1,out2)
            
            train_loss = train_loss1 + train_loss2 + (lam * gamma * D_ec).sum()
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_sum += train_loss.item()
            train_acc_sum += (out1.argmax(dim=1) == y1).sum().item()
            train_acc_sum += (out2.argmax(dim=1) == y2).sum().item()
            
            n += y.size(0)

            # break
        val_acc = evaluate_accuracy(dataloaders['valid'], net, device)
        print('epoch % d, loss % .4f, train acc % .3f, test acc % .3f, time % .1f sec'% (epoch + 1, train_loss_sum / n, train_acc_sum / n, val_acc, time.time() - start))
        if best_acc<val_acc:
            best_acc = val_acc
            time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
            torch.save(net.state_dict(), os.path.join(path,'best_model.pth'.format(time_stamp)))
            print('Saved model: ',time_stamp)