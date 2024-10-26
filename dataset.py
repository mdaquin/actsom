
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
        IS = []
        OS = []
        # CS = []
        for d in data:
            IS.append(d["I"])
            OS.append(d["O"])
            # CS.append(d["C"])
        IS = torch.Tensor(IS)
        OS = torch.Tensor(OS)
        # CS = torch.Tensor(CS) // TODO: will have to deal with this at some point
        return IS, OS
    
    def __iter__(self):
        self._ni = 0
        return self
    
    def __next__(self):
        if self._ni >= len(self.flist): StopIteration
        else:
            ret = self[self._ni]
            self._ni += 1
            return ret