from argparse import ArgumentParser
import numpy as np
import json, sys, pygame, time, torch
import importlib as imp
import utils as u
import matgl
from matgl.utils.training import ModelLightningModule
import pickle

parser = ArgumentParser(prog="Activation datasets", description="Create subset of high and low activations for cells of a SOM")
parser.add_argument('configfile')
parser.add_argument('layer')

args = parser.parse_args()
config = json.load(open(args.configfile))
actsom_dir = config["actsomPDSdir"]

torch.manual_seed(config["seed"])
base_sae_dir = config["saedir"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "runcpu" in config: device = torch.device("cpu")
if device == torch.device("cuda"): print("USING GPU")

### special megnet
with torch.no_grad():
    model = matgl.load_model("megnet/model")

# load the som
layer = args.layer
print("Loading SAE for layer", args.layer)
sae = torch.load(config["saedir"]+"/"+layer+".pkl", weights_only=False)

print("Setting up activation hooks...")
u.activation = {}
list_layers = u.set_up_activations(model)

print("Loading dataset...")
# spec = imp.util.spec_from_file_location(config["datasetmodulename"], config["datasetclass"])
# module = imp.util.module_from_spec(spec)
# sys.modules[config["datasetmodulename"]] = module
# spec.loader.exec_module(module)
# exec("import "+config["datasetmodulename"])
# ### special megnet
# loader1, loader2, loader3 = eval(config["datasetcode"])

# loading datasets
with open("megnet/data/mp.2018.6.1_structures.pkl", "rb") as f:
        structures = pickle.load(f)
with open("megnet/data/mp.2018.6.1_mp_ids.pkl", "rb") as f:
        mp_ids = pickle.load(f)
# with open("megnet/data/mp.2018.6.1_eform_per_atom.pkl", "rb") as f:
#         eform_per_atom = pickle.load(f)
 
def insert(rank, value, tables, n):
    found=False
    for i in range(len(tables["hc"])):
        if value > tables["ha"][i]:
              # insert v in the tables["ha"] list at rank i
              tables["hc"].insert(i, rank)
              tables["ha"].insert(i, value)
              found=True
              break
    if len(tables["hc"]) < n and not found:
          tables["hc"].append(rank)
          tables["ha"].append(value)
    tables["hc"] = tables["hc"][:n]
    tables["ha"] = tables["ha"][:n]
    found=False
    for i in range(len(tables["lc"])):
        if value < tables["la"][i]:
              # insert v in the tables["la"] list at rank i
              tables["lc"].insert(i, rank)
              tables["la"].insert(i, value)
              found=True
              break
    if len(tables["lc"]) < n and not found:
            tables["lc"].append(rank)
            tables["la"].append(value)
    tables["lc"] = tables["lc"][:n]
    tables["la"] = tables["la"][:n]
 

def get_activations_megnet(t):
        if type(t) == tuple and len(t) > 1: t = t[0].detach() # only the first one for now...
        if type(t) == tuple and len(t) > 1: t = t[0].detach() # can be a tuple in a tuple
        if len(t.shape) == 1: return t.detach()
        if len(t.shape) == 2:
            if t.shape[0] == 1: return t[0]
            else: return torch.mean(t, dim=1).detach() # could be other aggregation methods
        if len(t.shape) == 3:
            if t.shape[0] == 1 and t.shape[1] == 1: return t[0][0].detach()
            else: return torch.mean(torch.mean(t, dim=2), dim=1).detach() # randomly, I don't think this happens
        # print("!!!!!!!!!!!!", len(t.shape))
        return None

oacts = None
result = None
print("Iterating over dataset")
# this would be more efficient with batching 
with torch.no_grad():
  #### special megnet
  for i, struct in enumerate(structures):
    if "mp-" not in mp_ids[i]: continue
    # print(mp_ids[i])
    u.activation = {}
    pred = model.predict_structure(struct)   
    acts = get_activations_megnet(u.activation[layer])
    # print(acts.shape)
    # acts = (acts-som.minval)/(som.maxval-som.minval) # we don't do this for SAE combination...
    acts = torch.unsqueeze(acts, 0).to(device)
    decoded, encoded = sae(acts)
    if result is None: result = [{"hc":[], "ha":[], "lc":[], "la": []} for i in range(len(encoded[0]))]
    for j,v in enumerate(encoded[0]):
           insert(i, float(v), result[j], config["Nacts"])
    if i%100 == 0: print(".", end="")
print()

print(result)
with open(actsom_dir+"/"+layer+"_sae.json", "w") as f:
    json.dump(result, f)