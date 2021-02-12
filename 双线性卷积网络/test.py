from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from base_model import Net
import torch
from torchvision import transforms

tr = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = torch.load('./2021-02-12-18-38-35-best_model.pth')

net = Net(130)
net.load_state_dict(model)

net.eval()

pre = {}

test_path = 'E:\\Privatedocuments\\BDC\\jittor\\dataset\\TEST_A\\'
for img in tqdm(os.listdir(test_path)):

    image = Image.open(test_path+img).convert('RGB')
    
    image = tr(image)
    
    out = net(torch.from_numpy(np.array([image.numpy()])))
    out = out.detach().numpy()
    top5 = np.argsort(-out,axis=1)[0][:5]
    pre.update({
        img:list(top5)
    })

for k,v in pre.items():
    pre[k] = [ int(v[i])+1 for i in range(len(v))]
    
item = json.dumps(pre)
with open('./result.json','w') as f:
    f.write(item)
