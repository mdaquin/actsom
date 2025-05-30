import math
import pandas as panda
import torch, sys, os, json
from ksom import SOM, cosine_distance, nb_gaussian, nb_linear, nb_ricker
import importlib as imp
import utils as u


class SpecialDataLoader:

    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
    
    def __len__(self):
        return math.ceiling(len(self.data)/self.batch_size)
    
    def __getitem__(self, idx):
        return self.data[idx*self.batch_size:(idx+1)*self.batch_size]

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("provide configuration file (JSON)")
        sys.exit(1)
    else: conf = sys.argv[1]

    # base config
    config = json.load(open(conf))
    torch.manual_seed(config["seed"])
    som_size = config["som_size"]
    base_som_dir = config["somdir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runcpu = "runcpu" in config and config["runcpu"]
    if runcpu: device = torch.device("cpu")
    if device == torch.device("cuda"): print("USING GPU")
       
    with torch.no_grad():
     print("Loading model...")   
     # exec(open(config["modelclass"]).read())
     spec = imp.util.spec_from_file_location(config["modelmodulename"], config["modelclass"])
     module = imp.util.module_from_spec(spec)
     sys.modules[config["modelmodulename"]] = module
     spec.loader.exec_module(module)
     exec("import "+config["modelmodulename"])
     if "model" in config: model = u.load_model(config["model"], device=device)
     else: model = eval(config["modelcode"]) 
     model.eval()

    print("Loading dataset...")
    spec = imp.util.spec_from_file_location(config["datasetmodulename"], config["datasetclass"])
    module = imp.util.module_from_spec(spec)
    sys.modules[config["datasetmodulename"]] = module
    spec.loader.exec_module(module)
    exec("import "+config["datasetmodulename"])
    data = eval(config["datasetcode"])
    special_dataloader = "specialdataloader" in config and config["specialdataloader"]
    if not special_dataloader:
        data_loader = torch.utils.data.DataLoader(data, batch_size=config["batchsize"], shuffle=True)
    else: 
        data_loader=SpecialDataLoader(data, batch_size=config["batchsize"])

    print("Setting up activation hooks...")
    u.activation = {}
    list_layers = u.set_up_activations(model)
    print(list_layers)

    print("Training SOMs")
    SOMs = {}
    mm = {}
    SOMevs = {}
    with torch.no_grad(): # SOM training does not require gradients
     for ep in range(1, config["nepochs"]+1):
        count=0
        sev  = 0
        for X, y in data_loader:
            if not runcpu: X = X.to(device)
            if not runcpu: y = y.to(device)
            # print("   ** applying model")
            u.activation = {}
            p = model(X)
            if config["eval"] == "precision":
                # multiclass
                if len(y.shape) > 1:
                    y = torch.argmax(y, dim=1)
                    p = torch.argmax(p, dim=1)
                    sev += (y==p).sum()/len(p)
                else: sev += 1-(abs(y - (p>=0.5).to(torch.int).T[0]).sum()/len(p))
            elif config["eval"] == "mae":
                sev += abs(y - p.T[0]).sum()/len(p) 
            elif config["eval"] == "mse":
                sev += ((y - p.T[0])**2).sum()/len(p) 
            count+=1
            for layer in u.activation:
                # in case of LSTM, it is a tuple
                # output is the first element
                if type(u.activation[layer]) == tuple: u.activation[layer] = u.activation[layer][0]
                # dealing with MegNet weird activation shapes
                if len(u.activation[layer].shape) < 2: continue
                if not u.activation[layer].shape[0] == config["batchsize"]:
                    if u.activation[layer].shape[1] == config["batchsize"]:
                        u.activation[layer] = u.activation[layer].T
                    else: continue
                if config["aggregation"] == "flatten": acts = torch.flatten(u.activation[layer], start_dim=1).to(device)
                elif config["aggregation"] == "mean":
                    if len(u.activation[layer].shape) > 2:
                        acts = torch.mean(u.activation[layer], dim=1).to(device)
                    else: acts = u.activation[layer].to(device)
                else: 
                    print("unknown aggregation, check config")
                    sys.exit(-1)
                if acts.shape[1] < 10: continue
                # print("acts.shape", acts.shape)
                if layer not in mm:
                    mm[layer] = {
                        "min": acts.min(dim=0).values.to(device),
                        "max": acts.max(dim=0).values.to(device)
                        }
                # normalisation based on min/max of first dataset
                if acts.shape[1] != mm[layer]["min"].shape[0] or acts.shape[1] != mm[layer]["max"].shape[0]: 
                    #if layer in SOMs: 
                        #print("*** dropping SOM", layer)
                        #del SOMs[layer]
                    continue                
                print("*** progressing with", layer)
                acts = (acts-mm[layer]["min"])/(mm[layer]["max"]-mm[layer]["min"])
                if layer not in SOMs and len(acts.shape) == 2: # how can it not be?
                  print("   ** creating", layer)
                  perm = torch.randperm(acts.size(0))
                  samples = acts[perm[-(som_size[0]*som_size[1]):]]
                  SOMs[layer] = SOM(som_size[0], 
                                  som_size[1], 
                                  acts.shape[1], 
                                  dist=cosine_distance,
                                  neighborhood_init=som_size[0]*1.0, 
                                  neighborhood_drate=0.00001*som_size[0], 
                                  zero_init=True,
                                  sample_init=samples,
                                  minval=mm[layer]["min"], 
                                  maxval=mm[layer]["max"], 
                                  device=device, 
                                  alpha_init=config["alpha"],
                                  neighborhood_fct=nb_linear, 
                                  alpha_drate=config["alpha_drate"])
                  SOMevs[layer] = {"change": 0.0, "count": 0}
                if layer not in SOMs: continue
                change,count2 = SOMs[layer].add(acts.to(device))
                SOMevs[layer]["change"] += change
                SOMevs[layer]["count"] += count2
                # NaNs happen quickly, from first relu layer.
                # this is a trick... should be investigated why we get NaNs in the SOM
                if torch.isnan(SOMs[layer].somap).any(): 
                    print ("*** NaN!")
                    SOMs[layer].somap = torch.nan_to_num(SOMs[layer].somap, 0.0)   
        print(f"{ep}:: Model eval={sev/count}, mem use: {torch.cuda.memory_allocated('cuda:0')/(1014**3):.2f}GB")
        for layer in SOMevs:
            SOMevs[layer]["change"] /= count
            SOMevs[layer]["count"] /= count
            print(f"    {layer}:: change={SOMevs[layer]['change']}, count={SOMevs[layer]['count']}")
            # save SOMs     
            torch.save(SOMs[layer], base_som_dir+"/"+layer+".pt")
    
