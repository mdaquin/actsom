import os
import sys
import torch 
import utils as u
import json
import pandas as pd

if len(sys.argv) != 2: 
    print("Please provide a config file.")
    sys.exit(-1)

config = json.load(open(sys.argv[1]))

print("*** loading model")
exec(open(config["modelclass"]).read())
model=u.load_model(config["model"])

global activation
activation = {}
def get_activation(name):
    def hook(model, input, output):
            if type(output) != torch.Tensor: activation[name] = output
            else: activation[name] = output.cpu().detach()
    return hook
soms = os.listdir(config["base_som_dir"])
for som in soms:
    layer = som
    if "/" in layer: layer = layer[layer.rindex("/")+1:]
    if ".pt" in layer: layer = layer[:layer.rindex(".")]
    print("*** setup hook for layer", layer)
    smod = u.getLayer(model, layer)
    smod.register_forward_hook(get_activation(layer))

results = {"target": [], "pred": []}
saeactivations = {}
dataset = u.KSDataset(config["dataset_dir"], return_c=True)
for i in range(len(dataset)):
    print("   *** file", i)
    IS,OS,CS, = dataset[i] # add the concetps
    df1 = pd.DataFrame(CS)
    for col in df1:
        if col not in results: results[col] = []
        results[col] += list(df1[col])
    if IS.to(int).equal(IS): IS = IS.to(int)
    results["target"] += list(OS.detach().numpy())
    PS = model(IS)
    results["pred"] += list(PS.detach().numpy())
    soms = os.listdir(config["base_som_dir"])
    for somf in soms:
        layer = somf
        if "/" in layer: layer = layer[layer.rindex("/")+1:]
        if ".pt" in layer: layer = layer[:layer.rindex(".")]
        print("      *** applying SOM for",layer)
        som = torch.load(config["base_som_dir"]+"/"+somf, weights_only=False) # should rather keep in memory ? 
        som.to("cpu")
        if type(activation[layer]) == tuple: activation[layer] = activation[layer][0]
        if config["aggregation"] == "flatten":
            acts = torch.flatten(activation[layer], start_dim=1).cpu()
        elif config["aggregation"] == "mean":
            if len(activation[layer].shape) > 2:
                acts = torch.mean(activation[layer], dim=1).cpu()
            else: 
                acts = activation[layer].cpu()
        else:
            print("unknown aggregation")
            sys.exit(-1)
        acts = (acts-som.minval.cpu())/(som.maxval.cpu()-som.minval.cpu())
        res = som(acts)[0]
        res = res.numpy().T
        res = res[0]*som.xs+res[1]
        # res = [layer+"_"+str(x) for x in res]
        if layer not in results: results[layer] = []
        results[layer] += list(res)
        sae = config["base_sae_dir"]+"/"+layer+".pt"
        if not os.path.exists(sae): continue
        print("      *** applying SAE for",layer)
        sae = torch.load(sae, weights_only=False)
        sae.to("cpu")
        res = sae(acts)[1]
        if layer not in saeactivations: saeactivations[layer] = []
        saeactivations[layer].append(res.detach().numpy().tolist())
df = pd.DataFrame(results)
df.to_csv(config["som_results_file"])
for layer in saeactivations:
    json.dump(saeactivations[layer], open(config["sae_results_dir"]+layer+".json", "w"))