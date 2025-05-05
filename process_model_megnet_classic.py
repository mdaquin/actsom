import math
import pandas as panda
import torch, sys, os, json
from ksom import SOM, cosine_distance, nb_gaussian, nb_linear, nb_ricker
import importlib as imp
import utils as u
from matgl.utils.training import ModelLightningModule
import matgl
import pickle

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
        model = matgl.load_model("megnet/model")

    print(model)

    print("Loading dataset...")
    spec = imp.util.spec_from_file_location(config["datasetmodulename"], config["datasetclass"])
    module = imp.util.module_from_spec(spec)
    sys.modules[config["datasetmodulename"]] = module
    spec.loader.exec_module(module)
    exec("import "+config["datasetmodulename"])
    #### special megnet
    loader1, loader2, loader3 = eval(config["datasetcode"])

    # loading datasets
    with open("megnet/data/mp.2018.6.1_structures.pkl", "rb") as f:
        structures = pickle.load(f)
    with open("megnet/data/mp.2018.6.1_mp_ids.pkl", "rb") as f:
        mp_ids = pickle.load(f)
    with open("megnet/data/mp.2018.6.1_eform_per_atom.pkl", "rb") as f:
        eform_per_atom = pickle.load(f)

    print("Setting up activation hooks...")
    u.activation = {}
    list_layers = u.set_up_activations(model)
    print(list_layers)

    def get_activations_megnet(t):
        if type(t) == tuple and len(t) > 1: t = t[0] # only the first one for now...
        if type(t) == tuple and len(t) > 1: t = t[0] # can be a tuple in a tuple
        if len(t.shape) == 1: return t
        if len(t.shape) == 2:
            if t.shape[0] == 1: return t[1]
            else: return torch.mean(t, dim=1) # could be other aggregation methods
        if len(t.shape) == 3:
            if t.shape[0] == 1 and t.shape[1] == 1: return t[2]
            else: return torch.mean(torch.mean(t, dim=2), dim=1) # randomly, I don't think this happens
        print("!!!!!!!!!!!!", len(t.shape))
           
            #     if 2D, and shape[0] != 1, mean (or sum, or max, or ?)
            #     if 3D and shape[0] = shape[1] = 1, take dimension 2,


    print("Training SOMs")
    SOMs = {}
    mm = {}
    SOMevs = {}
    with torch.no_grad(): # SOM training does not require gradients
      for ep in range(1, config["nepochs"]+1):
        count=0
        sev  = 0
        for i, struct in enumerate(structures):
            print(i, "*******************************")
            u.activation = {}
            pred = model.predict_structure(struct)        
            for layer in u.activation:
                acts = get_activations_megnet(u.activation[layer])
                print(layer, "::", acts.shape)
            # create a batch of batch_size
            if i == 2: break
                # if layer not in mm:
                #     mm[layer] = {
                #         "min": acts.min(dim=0).values.to(device),
                #         "max": acts.max(dim=0).values.to(device)
                #         }
                # normalisation based on min/max of first dataset
                # if acts.shape[0] < som_size[0]*som_size[1]: 
                #     print("######### not enought samples", layer)
                #     continue
                # if acts.shape[1] != mm[layer]["min"].shape[0] or acts.shape[1] != mm[layer]["max"].shape[0]: 
                #     print("################### problem with sizes", layer)
                #     #if layer in SOMs: 
                #         #print("*** dropping SOM", layer)
                #         #del SOMs[layer]
                #     continue                
        #         acts = (acts-mm[layer]["min"])/(mm[layer]["max"]-mm[layer]["min"])
        #         if layer not in SOMs and len(acts.shape) == 2: # how can it not be?
        #           print("   ** creating", layer)
        #           perm = torch.randperm(acts.size(0))
        #           samples = acts[perm[-(som_size[0]*som_size[1]):]]
        #           SOMs[layer] = SOM(som_size[0], 
        #                           som_size[1], 
        #                           acts.shape[1], 
        #                           dist=cosine_distance,
        #                           neighborhood_init=som_size[0]*1.0, 
        #                           neighborhood_drate=0.00001*som_size[0], 
        #                           zero_init=True,
        #                           sample_init=samples,
        #                           minval=mm[layer]["min"], 
        #                           maxval=mm[layer]["max"], 
        #                           device=device, 
        #                           alpha_init=config["alpha"],
        #                           neighborhood_fct=nb_linear, 
        #                           alpha_drate=config["alpha_drate"])
        #           SOMevs[layer] = {"change": 0.0, "count": 0}
        #         if layer not in SOMs: continue
        #         change,count2 = SOMs[layer].add(acts.to(device))
        #         SOMevs[layer]["change"] += change
        #         SOMevs[layer]["count"] += count2
        #         # NaNs happen quickly, from first relu layer.
        #         # this is a trick... should be investigated why we get NaNs in the SOM
        #         if torch.isnan(SOMs[layer].somap).any(): 
        #             print ("*** NaN!")
        #             SOMs[layer].somap = torch.nan_to_num(SOMs[layer].somap, 0.0)   
        #         count+=1
        # #print(f"{ep}:: Model eval={sev/count}, mem use: {torch.cuda.memory_allocated('cuda:0')/(1014**3):.2f}GB")
        # for layer in SOMevs:
        #     SOMevs[layer]["change"] /= count
        #     SOMevs[layer]["count"] /= count
        #     print(f"  - {layer}:: change={SOMevs[layer]['change']}, count={SOMevs[layer]['count']}")
        #     # save SOMs     
        #     torch.save(SOMs[layer], base_som_dir+"/"+layer+".pt")
    
