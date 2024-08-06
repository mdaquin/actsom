import pandas as panda
import torch
import sys
import json

## NEXT STEPS:: ##
# Way to go through each layer
# Create dataset as csv
# Run training sets
# Get activations of each layer... if conv2D, concat matrix of each channel separetly
# Add to SOM for each layer

def load_model(fn, device="cpu"):
    return torch.load(fn).to(device)

def set_up_activations(model):
    global activation 
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
            rec_reg_hook(nmo, prev=name, lev=lev+1)
    rec_reg_hook(model)

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
    set_up_activations(model)
