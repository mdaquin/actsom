import pandas as panda
import torch
import sys, os
import json
from ksom import SOM, cosine_distance, nb_gaussian, nb_linear, nb_ricker

# TODO figure out the NaNs in SOMs

def load_model(fn, device="cpu"):
    return torch.load(fn, map_location=device, weights_only=False)

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
    
if __name__ == "__main__":

    if len(sys.argv) != 2:
        conf = "config_painters.json"
        #print("provide configuration file (JSON)")
        #sys.exit(-1)
    else: conf = sys.argv[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"): print("USING GPU")
        
    config = json.load(open(conf))
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
        OS = []
        for d in data: 
            IS.append(d["I"])
            OS.append(d["O"])
        IS = torch.Tensor(IS)
        OS = torch.Tensor(OS)
        if "inputisint" in config and config["inputisint"] == 1: IS = IS.to(torch.int)
        IS = IS.to(device)
        OS = OS.to(device)
        print("*** applying model ***")
        activation = {}
        P = model(IS)
        if config["eval"] == "precision":
            prec = 1-(abs(OS - (P>=0.5).to(torch.int).T[0]).sum()/len(P))
            print(f" Precision: {prec*100:.02f}%")
        elif config["eval"] == "mae":
            err = abs(OS - P.T[0]).sum()/len(P) 
            print(f" Average error: {err:.02f}")
        for layer in activation:
            # in case of LSTM, it is a tuple
            # output is the first element
            if type(activation[layer]) == tuple: activation[layer] = activation[layer][0]
            if config["aggregation"] == "flatten": acts = torch.flatten(activation[layer], start_dim=1).to(device)
            elif config["aggregation"] == "mean":
                if len(activation[layer].shape) > 2:
                    acts = torch.mean(activation[layer], dim=1).to(device)
                else: acts = activation[layer].to(device)
            else: 
                print("unknown aggregation, check config")
                sys.exit(-1)
            if layer not in mm:
                mm[layer] = {
                    "min": acts.min(dim=0).values.to(device),
                    "max": acts.max(dim=0).values.to(device)
                    }
            # normalisation based on min/max of first dataset
            acts = (acts-mm[layer]["min"])/(mm[layer]["max"]-mm[layer]["min"])
            if layer not in SOMs: 
                print("      *** creating", layer)
                SOMs[layer] = SOM(som_size[0], 
                                  som_size[1], 
                                  acts.shape[1], 
                                  dist=cosine_distance,
                                  neighborhood_init=som_size[0]*2.0, 
                                  neighborhood_drate=0.00001*som_size[0], 
                                  zero_init=True,
                                  minval=mm[layer]["min"], 
                                  maxval=mm[layer]["max"], 
                                  device=device, 
                                  alpha_init=config["alpha"],
                                  neighborhood_fct=nb_linear, 
                                  alpha_drate=config["alpha_drate"])
            print("   *** adding to SOM for",layer)
            change,count = SOMs[layer].add(acts.to(device))
            print(f"      {count}/{len(acts)} elements resulted in a change of {change}")
            torch.save(SOMs[layer], base_som_dir+"/"+layer+".pt")
            # NaNs happen quickly, from first relu layer.
            # this is a trick... should be investigated why we get NaNs in the SOM
            if torch.isnan(SOMs[layer].somap).any(): 
                print ("*** NaN!")
                SOMs[layer].somap = torch.nan_to_num(SOMs[layer].somap, 0.0)   
        print("*** done ***")
