
import json
import os

import torch

class KSDataset : 

    def __init__(self, data_dirname):
        self.dirname = data_dirname
        self.flist = os.listdir(self.dirname)
    
    def __len__(self): return len(self.flist)

    def __getitem__(self, i): 
        data = json.load(open(self.dirname+"/"+self.flist[i]))
        IS, OS = [],[]
        for item in data:
            IS.append(item["I"])
            OS.append(item["O"])
        IS = torch.Tensor(IS)
        OS = torch.Tensor(OS)
        # CS = torch.Tensor(CS) // TODO: will have to deal with this at some point
        return IS, OS
