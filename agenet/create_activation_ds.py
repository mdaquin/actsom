import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

dataset_path = "/data/data/ext_datasets/UTKFace/"
image_size = (200,200)
batch_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"): print("USING GPU")

class UTKFace(Dataset):
    def __init__(self, image_paths, dataset_path):
        self.transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.image_paths = image_paths
        self.dataset_path = dataset_path
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []
        
        for path in image_paths:
            filename = path[:].split("_")
            if len(filename)==4:
                self.images.append(path)
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))

    def __len__(self):
         return len(self.images)

    def __getitem__(self, index):
            img = Image.open(self.dataset_path+self.images[index]).convert('RGB')
            img = self.transform(img)          
            age = self.ages[index]
            gender = self.genders[index]
            eth = self.races[index]
            return {"img":img, "age":age, "gender": gender, "ethnicity": eth}

files = os.listdir(dataset_path)
dataloader = DataLoader(UTKFace(files, dataset_path), shuffle=False, batch_size=batch_size)

class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        
        self.linear = nn.Linear(self.n_features,self.n_features)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.n_features, 1)
        
    def forward(self, x):
        out = self.net(x)
        out = self.linear(out)
        out = self.relu(out)
        out = self.out(out)
        return out

model = torch.load("agenet_6.44_0.8").to(device=device)

def get_activation(name):
     def hook(model, input, output):
         if type(output) != torch.Tensor: activation[name] = output
         else: activation[name] = output.cpu().detach()
     return hook

for k in model.__dict__["_modules"]:
    print("=== hook for ",k,"===")
    getattr(model,k).register_forward_hook(get_activation(k))
    for kk in getattr(model,k).__dict__["_modules"]:
        print("  hook for",k+"."+kk)
        getattr(getattr(model,k),kk).register_forward_hook(get_activation(k+"."+kk))

X=[]
y=[]
c=[]
act={}

limit = 5000

model.eval()
count=0
nl = 1
with torch.no_grad():    
    for d in dataloader:
        print(count*batch_size)
        count+=1        
        if count*batch_size > limit:
            torch.save(X,"agenet_X_"+str(nl))
            torch.save(y,"agenet_y_"+str(nl))
            torch.save(c,"agenet_c_"+str(nl))
            torch.save(act,"agenet_act_"+str(nl))
            X=[]
            y=[]
            c=[]
            act={}
            nl+=1
            count=0
        activation = {}
        out = model(d["img"].to(device))
        d["img"] = d["img"].cpu().detach()
        for i,v in enumerate(d["img"]):
            X.append(v)
            y.append(d["age"][i])
            c.append({"gender": d["gender"][i], "ethnicity": d["ethnicity"][i]})
            for k in activation:
                if k not in act: act[k]=[]
                act[k].append(activation[k][i])                


torch.save(X,"agenet_X_"+str(nl))
torch.save(y,"agenet_y_"+str(nl))
torch.save(c,"agenet_c_"+str(nl))
torch.save(act,"agenet_act_"+str(nl))
