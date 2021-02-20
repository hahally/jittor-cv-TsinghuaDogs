from numpy.core.numeric import indices
import torch
import torch.nn as nn
from torchvision.models import resnet18,resnet50,vgg16,vgg16_bn
import torch.nn.functional as F

class HBPCNN(nn.Module):
    def __init__(self, n_classes,pretrained=False):
        super(HBPCNN, self).__init__()
        pretrained_vgg = vgg16(pretrained=pretrained)
        if pretrained == True:
            for param in pretrained_vgg.parameters():
                param.requires_grad = False
                
        self.features = pretrained_vgg.features
        
        self.feat1 = self.features[:-5]
        self.feat2 = self.features[-5:-3]
        self.feat3 = self.features[-3:-1] 

        self.classifiers = nn.Sequential(nn.Linear(512*512*3, n_classes))
     
    
    def bpool(self, x,y):
        batch_size = x.size(0)
        feature_size = x.size(2) * x.size(3)
        
        x = x.view(batch_size, 512, feature_size)
        y = y.view(batch_size, 512, feature_size)
        
        x = (torch.bmm(x, torch.transpose(y, 1, 2)) / feature_size).view(
            batch_size, -1)
        x = torch.nn.functional.normalize(
            torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        
        return x    

    def forward(self,x):
        x1 = self.feat1(x)
        x2 = self.feat2(x1)
        x3 = self.feat3(x2)
        
        y1 = self.bpool(x3,x1)
        y2 = self.bpool(x3,x2)
        y3 = self.bpool(x3,x3)
        
        y = torch.cat([y1,y2,y3],dim=1)
        
        out = self.classifiers(y)
        
        return out

class DFL_VGG16(nn.Module):
	def __init__(self, k = 10, n_classes = 130, pretrained = False):
		super(DFL_VGG16, self).__init__()
		self.k = k
		self.nclass = n_classes
		
		# k channels for one class, nclass is total classes, therefore k * nclass for conv6
		vgg16featuremap = vgg16_bn(pretrained=pretrained).features
		conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-11])
		conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-11:])
		conv6 = torch.nn.Conv2d(512, k * n_classes, kernel_size = 1, stride = 1, padding = 0)
		pool6 = torch.nn.MaxPool2d((56, 56), stride = (56, 56), return_indices = True)

		# Feature extraction root
		self.conv1_conv4 = conv1_conv4

		# G-Stream
		self.conv5 = conv5
		self.cls5 = nn.Sequential(
			nn.Conv2d(512, 200, kernel_size=1, stride = 1, padding = 0),
			nn.BatchNorm2d(200),
			nn.ReLU(True),
			nn.AdaptiveAvgPool2d((1,1)),
			)

		# P-Stream
		self.conv6 = conv6
		self.pool6 = pool6
		self.cls6 = nn.Sequential(
			nn.Conv2d(k * n_classes, n_classes, kernel_size = 1, stride = 1, padding = 0),
			nn.AdaptiveAvgPool2d((1,1)),
			)
		# Side-branch
		self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)

	def forward(self, x):
		batchsize = x.size(0)

		# Stem: Feature extraction
		inter4 = self.conv1_conv4(x)

        # G-stream
		x_g = self.conv5(inter4)
		out1 = self.cls5(x_g)
		out1 = out1.view(batchsize, -1)

		# P-stream ,indices is for visualization
		x_p = self.conv6(inter4)
		x_p, indices = self.pool6(x_p)
		inter6 = x_p
		out2 = self.cls6(x_p)
		out2 = out2.view(batchsize, -1)
		
		# Side-branch
		inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
		out3 = self.cross_channel_pool(inter6)
		out3 = out3.view(batchsize, -1)
	
		return out1, out2, out3, indices
  
  

