from argparse import ArgumentParser
import numpy as np
import json, sys, pygame, time, torch
import importlib as imp
import utils as u

parser = ArgumentParser(prog="Activation datasets", description="Create subset of high and low activations for cells of a SOM")
parser.add_argument('configfile')
parser.add_argument('layer')
parser.add_argument('-o', '--output') # output json file

args = parser.parse_args()
config = json.load(open(args.configfile))

torch.manual_seed(config["seed"])
som_size = config["som_size"]
base_som_dir = config["somdir"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "runcpu" in config: device = torch.device("cpu")
if device == torch.device("cuda"): print("USING GPU")

with torch.no_grad():
    print("Loading model...")   
    # exec(open(config["modelclass"]).read())
    spec = imp.util.spec_from_file_location(config["modelmodulename"], config["modelclass"])
    module = imp.util.module_from_spec(spec)
    sys.modules[config["modelmodulename"]] = module
    spec.loader.exec_module(module)
    model = u.load_model(config["model"], device=device)
    model.eval()

# load the som
layer = args.layer
print("Loading SOM for layer", args.layer)
som = torch.load(config["somdir"]+"/"+layer+".pt", weights_only=False)

print("Setting up activation hooks...")
u.activation = {}
list_layers = u.set_up_activations(model)

print("Loading dataset...")
spec = imp.util.spec_from_file_location(config["datasetmodulename"], config["datasetclass"])
module = imp.util.module_from_spec(spec)
sys.modules[config["datasetmodulename"]] = module
spec.loader.exec_module(module)
exec("import "+config["datasetmodulename"])
data = eval(config["datasetcode"])


def insert(rank, value, tables, n):
    found=False
    for i in range(len(tables["hc"])):
        if v > tables["ha"][i]:
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
        if v < tables["la"][i]:
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
 
result = [{"hc":[], "ha":[], "lc":[], "la": []} for i in range(som_size[0]*som_size[1])]
print("Iterating over dataset")
# this would be more efficient with batching 
with torch.no_grad():
  for i,(X,y) in enumerate(data):
    u.activation = {}
    X = torch.unsqueeze(X, 0).to(device)
    p = model(X)
    acts = u.activation[layer]
    if type(acts) == tuple: acts = acts[0]
    if config["aggregation"] == "flatten": 
            acts = torch.flatten(acts, start_dim=1).to(device)
    elif config["aggregation"] == "mean":
                    if len(acts.shape) > 2:
                        acts = torch.mean(acts, dim=1).to(device)
                    else: acts = acts.to(device)
    else: 
            print("unknown aggregation, check config")
            sys.exit(-1)
    acts = (acts-som.minval)/(som.maxval-som.minval) # we don't do this for SAE combination...
    res = som(acts)
    for j,v in enumerate(res[1]):
          insert(i, float(v), result[j], config["Nacts"])
    if i%100 == 0: print(".", end="")
print()

print(result)
if "output" in args:
    with open(args.output, "w") as f:
        json.dump(result, f)