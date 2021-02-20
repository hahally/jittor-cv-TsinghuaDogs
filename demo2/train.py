import os
import time
from tqdm import tqdm
import torch



def evaluate_accuracy(data_iter, net, device):

    acc_sum, n = 0.0, 0
    net.eval()
    for x, y in tqdm(data_iter):
        x, y = x.to(device), y.to(device)
        out1,out2,out3,_ = net(x)
        out = out1 + out2 + 0.1 * out3
        acc_sum += (out.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train(net, optimizer, loss, scheduler,device, dataloaders, epochs=1, path='./'):

    best_acc = 0
    for epoch in range(epochs):
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        scheduler.step()
        net.train()
        for x, y in tqdm(dataloaders['train']):
            x, y = x.to(device), y.to(device)

            out1,out2,out3,_ = net(x)

            loss1 = loss(out1, y)
            loss2 = loss(out2, y)
            loss3 = loss(out3, y)
            
            train_loss = loss1 + loss2 + 0.1 * loss3
            out = out1 + out2 + 0.1 * out3
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_sum += train_loss.item()
            train_acc_sum += (out.argmax(dim=1) == y).sum().item()
            n += y.size(0)
  
            # break
        val_acc = evaluate_accuracy(dataloaders['valid'], net, device)
        print('epoch % d, loss % .4f, train acc % .3f, test acc % .3f, time % .1f sec'% (epoch + 1, train_loss_sum / n, train_acc_sum / n, val_acc, time.time() - start))
        if best_acc<val_acc:
            best_acc = val_acc
            time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
            torch.save(net.state_dict(), os.path.join(path,'best_model.pth'))
            print('Saved model: ',time_stamp)
            
    time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    torch.save(net.state_dict(), os.path.join(path,'{}-last_model.pth'.format(time_stamp)))
    