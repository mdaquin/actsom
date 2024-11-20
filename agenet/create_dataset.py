import os, sys, json
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms


if len(sys.argv) != 2:
    print("provide config file")
    sys.exit(-1)

config=json.load(open(sys.argv[1]))

dataset_path = config["path_to_data"]
image_size = tuple(config["image_size"])
batch_size=1024

class UTKFace(Dataset):
    def __init__(self, image_paths, dataset_path):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
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
            return {"img":img, "age":age, "gender": gender, "ethnicity": eth}

files = os.listdir(dataset_path)
dataloader = DataLoader(UTKFace(files, dataset_path), shuffle=False, batch_size=batch_size)

count=0

with torch.no_grad():    
    for d in dataloader:
        print("***",count*batch_size,"***")
        count+=1
        tosave = []        
        for i,v in enumerate(d["img"]):
           tosave.append({"I": v.tolist(),
                          "O": float(d["age"][i]),
                          "C": {"ethnicity":int(d["ethnicity"][i]), "gender": int(d["gender"][i])}})
        json.dump(tosave, open(f"dataset/dataset_{count}.json", "w"))
