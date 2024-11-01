
import json
import os

import torch

class KSDataset : 

    def __init__(self, data_dirname):
        self.dirname = data_dirname
        self.flist = os.listdir(self.dirname)
        self.data = []
        for f in self.flist: 
            self.data += json.load(open(self.dirname+"/"+f))
    
    def __len__(self): return len(self.data)

    def __getitem__(self, i): 
        data = self.data[i]
        IS = torch.Tensor(data["I"])
        OS = data["O"]
        # CS = torch.Tensor(CS) // TODO: will have to deal with this at some point
        return IS, OS
