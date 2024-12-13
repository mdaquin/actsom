
import json
import os

import torch

class KSDataset : 

    def __init__(self, data_dirname, return_c=False):
        self.dirname = data_dirname
        self.flist = os.listdir(self.dirname)
        self.return_c = return_c
    
    def __len__(self): return len(self.flist)

    def __getitem__(self, i): 
        data = json.load(open(self.dirname+"/"+self.flist[i]))
        IS, OS, CS = [],[],[]
        for item in data:
            IS.append(item["I"])
            OS.append(item["O"])
            CS.append(item["C"])
        IS = torch.Tensor(IS)
        OS = torch.Tensor(OS)
        # CS = torch.Tensor(CS) // TODO: will have to deal with this at some point
        if self.return_c: return IS, OS, CS
        return IS, OS

def load_model(fn, device="cpu"):
    return torch.load(fn, map_location=device, weights_only=False)

def getLayer(model, layer):
    if "." in layer:
        first = layer[:layer.index(".")]
        rest  = layer[layer.index(".")+1:]
        nmod = getattr(model, first)
        return getLayer(nmod, rest)
    return getattr(model, layer) # won't work with subsubmodule