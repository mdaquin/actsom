import torch, sys, json
from ksom import SOM, cosine_distance, euclidean_distance, nb_gaussian, nb_linear, nb_ricker
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("provide configuration file (JSON)")
        sys.exit(1)
    conf = sys.argv[1]

    # base config
    config = json.load(open(conf))
    if "seed" in config: torch.manual_seed(config["seed"])
    if "som_size" in config: som_size = config["som_size"]
    else:
        print("SOM size required in config")
        sys.exit(1)
    if "nepochs" in config: nepochs = config["nepochs"]
    else:
        print("nepochs required in config")
        sys.exit(1)
    if "batch_size" in config: batch_size = config["batch_size"]
    else: batch_size = 128
    if "distance" in config: distance = config["distance"]
    else: distance = "euclidean"
    if distance == "euclidean": distance = euclidean_distance
    elif distance == "cosine": distance = cosine_distance
    else: 
        print("unknwon distance", distance)
        sys.exit(1)
    if "alpha" in config: alpha=config["alpha"]
    else: alpha = 5e-3
    if "alpha_drate" in config: alpha_drate=config["alpha_drate"]
    else: alpha_drate = 1e-6
    if "neighb_func" in config: neighb_func = config["neighb_func"]
    else: neighb_func = "linear"
    if neighb_func == "linear": neighb_func=nb_linear
    elif neighb_func == "gaussian": neighb_func=nb_gaussian
    elif neighb_func == "ricker": neighb_func=nb_ricker
    else: 
        print("Unknown neighb_func", neighb_func)
        sys.exit(1)
    if "activationfile" in config: 
        data = torch.load(config["activationfile"])
        if "activation_field" in config: 
            if config["activation_field"] in data: activations = data[config["activation_field"]]
            else:
                print("activation_field", config["activation_field"], "not found in activation file.")
                sys.exit(1)
        else: 
            print("activation_field required in config")
    else: 
        print("activationfile or activationcode required in config (only activationfile implemented)")
        sys.exit(1)
    base_som_dir = config["somdir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runcpu = "runcpu" in config and config["runcpu"]
    if runcpu: device = torch.device("cpu")
    if device == torch.device("cuda"): print("USING GPU")
       
    for layer in activations: # make a train_som function and a train 1 epoch function
        with torch.no_grad(): # basic som training does not require gradient
            dataloader = DataLoader(activations[layer], batch_size=batch_size, shuffle=True)
            print()
            som = None
            for ep in range(1, nepochs+1):
                for acts in tqdm(dataloader, f"Training SOM for {layer}"):
                    if som is None:
                        print(acts)
                        sample = acts[:som_size[0]*som_size[1]]
                        mins = acts.min(dim=0).values.to(device)
                        maxs = acts.max(dim=0).values.to(device)
                        som = SOM(som_size[0], som_size[1], len(activations[layer][0]), 
                                  dist=distance,
                                  neighborhood_init=som_size[0]*1.0, 
                                  neighborhood_drate=0.00001*som_size[0], 
                                  zero_init=True, sample_init=sample,
                                  minval=mins, maxval=maxs, 
                                  alpha_init=alpha, alpha_drate=alpha_drate, 
                                  neighborhood_fct=neighb_func, 
                                  device=device)

    SOMs = {}
    mm = {}
    with torch.no_grad(): # SOM training does not require gradients
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
    
