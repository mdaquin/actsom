import os
import random
import time
import numpy as np

from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import models

from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error

dataset_path = "/data/data/ext_datasets/UTKFace/"
seed = 42
val_size = 0.2
batch_size = 128
n_epochs = 200
image_size = (200,200)

random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"): print("USING GPU")

class UTKFace(Dataset):
    def __init__(self, image_paths, dataset_path):
        self.transform = transforms.Compose([transforms.Resize(image_size), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])
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
            
            sample = {'image':img, 'age': age, 'gender': gender, 'ethnicity':eth}
            
            return sample

files = os.listdir(dataset_path)
random.shuffle(files)
valid_dataset=files[:round(val_size*len(files))]
train_dataset=files[round(val_size*len(files)):]
train_dataloader = DataLoader(UTKFace(train_dataset, dataset_path), shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(UTKFace(valid_dataset, dataset_path), shuffle=False, batch_size=len(valid_dataset))

class FaceNet(nn.Module):
    def __init__(self,headsize=None):
        super().__init__()
        self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print(self.net)
        self.n_features = self.net.fc.in_features
        self.headsize = headsize
        if headsize is None: self.headsize=self.n_features
        self.net.fc = nn.Identity()
        
        self.linear = nn.Linear(self.n_features, self.headsize)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.headsize, 1)
        
    def forward(self, x):
        out = self.net(x)
        out = self.linear(out)
        out = self.relu(out)
        out = self.out(out)
        return out

model = FaceNet(headsize=100).to(device=device)
criterion = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

minmae = 10

for epoch in range(n_epochs):
    t = time.time()
    model.train()
    total_training_loss = 0
    count=0
    tr_pred = []
    tr_targ = []
    for i, data in enumerate(train_dataloader):
        inputs = data["image"].to(device=device)
        age_label = data["age"].to(device=device)        
        optimizer.zero_grad()
        age_output = model(inputs)
        loss = criterion(age_output, age_label.unsqueeze(1).float())        
        loss.backward()
        optimizer.step()        
        total_training_loss += loss
        count+=1
        tr_pred.extend(age_output.cpu().detach().numpy())
        tr_targ.extend(age_label.cpu().detach().numpy())
    tr_r2 = r2_score(tr_targ,tr_pred)
    model.eval()
    with torch.no_grad():
        va_pred = []
        va_targ = []
        for i, data in enumerate(val_dataloader):
            inputs = data["image"].to(device=device)
            age_label = data["age"].to(device=device)        
            age_output = model(inputs)            
            va_pred.extend(age_output.cpu().detach().numpy())
            va_targ.extend(age_label.cpu().detach().numpy())
        va_mae = mean_absolute_error(va_targ,va_pred)
        va_mape = mean_absolute_percentage_error(va_targ, va_pred)
        va_r2 = r2_score(va_targ,va_pred)
        if va_mae < minmae:
            torch.save(model,f"agenet_{round(va_mae,2)}_{round(va_r2,2)}")            
    print(f"{epoch} ({round((time.time()-t)*1000)}ms) - loss: {round(float(total_training_loss)/count,2)}, train r2: {round(tr_r2,2)}, mae: {round(va_mae,2)}, mape: {round(va_mape, 2)}, r2: {round(va_r2,2)}")



