import torch
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

from load_data import LoadData
from base_model import Net
from train import train_model


def main():
    # ------------------ config ------------------ #
    n_classes = 130
    batch_size = 2
    epochs = 30
    save_model_path = './'
    img_size = 448
    # -------------------------------------------- #

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(img_size+50),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset = LoadData(data_dir='../dataset/JsonData/',
                          mode='train',
                          transforms=data_transforms['train'])

    train_loaders = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    dataset = LoadData(data_dir='../dataset/JsonData/',
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
    
    net = Net(n_classes=n_classes,pretrained=False).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.08, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    loss = torch.nn.CrossEntropyLoss(size_average=False)

    train_model(net, optimizer, loss, scheduler,device, dataloaders,epochs ,save_model_path)
    
    
if __name__=='__main__':
    main()