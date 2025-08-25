import torch, sys, json
sys.path.insert(0, "../KSOM/src/ksom")
from ksom import SOM
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

class ActNIdDS:
    def __init__(self, activations, ids):
        self.activations = activations
        self.ids = ids
    def __len__(self): return len(self.activations)
    def __getitem__(self, idx): return self.activations[idx], self.ids[idx]

def get_acts_som(som, dataloader):
    sacts = []
    sids = []
    with torch.no_grad():
        pbar = tqdm(dataloader, "Processing")
        for acts, ids in pbar:     
            acts = acts.to(device)
            acts = ((acts-som.minval)/(som.maxval-som.minval))
            bmu, dists = som(acts)
            sacts += dists.T.cpu().tolist()
            sids += ids
            dist = float(torch.tensor(sacts).min(dim=1).values.mean())
            pbar.set_postfix({'d': f'{dist:.8f}'})
    return sacts, sids

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("provide configuration file (JSON), a som file (Pickle) and an activation file (Pickle).")
        sys.exit(1)
    conf = sys.argv[1]
    somfile = sys.argv[2]

    som = torch.load(somfile, weights_only=False)

    # base config
    config = json.load(open(conf))
    if "batch_size" in config: batch_size = config["batch_size"]
    else: batch_size = 128

    data = torch.load(sys.argv[3], weights_only=False)
    if "activation_field" in config: 
        if config["activation_field"] in data: activations = data[config["activation_field"]]
        else:
            print("activation_field", config["activation_field"], "not found in activation file.")
            sys.exit(1)
    else: 
        print("activation_field required in config")
        sys.exit(1)
    if "ID_field" in config: 
        if config["ID_field"] in data: IDs = data[config["ID_field"]]
        else: 
            print("ID_field", config["ID_field"], "not found in activation file.")
            sys.exit(1)
    else: 
        print("ID_field required in config")
        sys.exit(1)    
    if "somact_dir" in config: somact_dir = config["somact_dir"]
    else: 
        print("somact_dir required in configuration")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runcpu = "runcpu" in config and config["runcpu"]
    if runcpu: device = torch.device("cpu")
    if device == torch.device("cuda"): print("USING GPU")
    som.to(device) # TODO: this should be fixed in ksom
    for layer in activations: 
         ds = ActNIdDS(activations[layer], IDs)
         dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
         sacts, sids = get_acts_som(som, dataloader)
         res = {"acts": sacts, "ids": sids}
         asomfile = somfile[somfile.rindex('/')+1:] if "/" in somfile else somfile
         torch.save(res, somact_dir+"/"+asomfile) 
