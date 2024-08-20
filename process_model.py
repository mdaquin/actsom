import pandas as panda
import torch
import sys, os
import json
from ksom import SOM

##### TODO
## really need some normalisation...
## based on mean of first dataset ?


def load_model(fn, device="cpu"):
    return torch.load(fn, map_location=device)

def set_up_activations(model):
    global activation
    llayers = []
    def get_activation(name):
        def hook(model, input, output):
            if type(output) != torch.Tensor: activation[name] = output
            else: activation[name] = output.cpu().detach()
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
    

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("provide configuration file (JSON)")
        sys.exit(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"): print("USING GPU")
        
    config = json.load(open(sys.argv[1]))
    exec(open(config["modelclass"]).read())
    model = load_model(config["model"], device=device)
    activation = {}
    list_layers = set_up_activations(model)

    som_size = config["som_size"]
    base_som_dir = config["base_som_dir"]
    
    SOMs = {}
    mm = {}

    data_dir = config["dataset_dir"]
    for f in os.listdir(data_dir):
        print("*** loading", f, "***")
        data = json.load(open(data_dir+"/"+f))
        IS = []
        for d in data:
            IS.append(d["I"])
        IS = torch.Tensor(IS).to(device)
        print("*** applying model ***")
        activation = {}
        P = model(IS)
        for layer in activation:
            acts = torch.flatten(activation[layer], start_dim=1)
            if layer not in mm:
                mm[layer] = {
                    "min": acts.min(),
                    "max": acts.max()
                    }
            acts = (acts-mm[layer]["min"])/(mm[layer]["max"]-mm[layer]["min"])
            if layer not in SOMs: SOMs[layer] = SOM(som_size[0], som_size[1], acts.shape[1], neighborhood_init=som_size[0]*2.0, neighborhood_drate=0.00001*som_size[0], minval=mm[layer]["min"], maxval=mm[layer]["max"])
            print("   *** adding to SOM for",layer)
            SOMs[layer].add(acts)
            torch.save(SOMs[layer], base_som_dir+"/"+layer+".pt")
            break
        print("*** done ***")
        break
