
import json # remove
import os # remove

import torch

# remove...
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

def list_layers(mo, prevlist=[], prev="", lev=0):
    for k in mo.__dict__["_modules"]:
            name = prev+"."+k if prev != "" else k
            nmo = getattr(mo,k)
            prevlist.append(name)
            list_layers(nmo, prevlist=prevlist, prev=name, lev=lev+1)
    return prevlist

activation = {}

def set_up_activations(model):
    global activation
    llayers = []
    def get_activation(name):
        def hook(model, input, output):
            if type(output) != torch.Tensor: activation[name] = output
            else: 
                activation[name] = output.cpu().detach()
                # print(name, activation[name].shape)
        return hook
    def rec_reg_hook(mo, prev="", lev=0):
        for k in mo.__dict__["_modules"]:
            name = prev+"."+k if prev != "" else k
            nmo = getattr(mo,k)
            nmo.register_forward_hook(get_activation(name))
            print("--"+"--"*lev, "hook added for",name)
            llayers.append(name)
            rec_reg_hook(nmo, prev=name, lev=lev+1)
        return llayers
    return rec_reg_hook(model)
    