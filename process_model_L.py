import torch
import sys, os
import json
from ksom import SOM, cosine_distance, nb_gaussian, nb_linear, nb_ricker
import numpy as np
from custom_functions import visualize_neuron_activity_all, check_neuron, plot_training_errors
import torch.nn.functional as F
from sparce_autoencoder import SparseAutoencoder, train_SparseAE
from torch.utils.data import Dataset, DataLoader

SEED = 43
torch.manual_seed(SEED)

# =============================================================================
# 
# =============================================================================

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



# =============================================================================
# 
# =============================================================================


    
if __name__ == "__main__":

    if len(sys.argv) != 2:
        conf = "config_painters.json"
        #print("provide configuration file (JSON)")
        #sys.exit(-1)
    else: conf = sys.argv[1]
    
    device = torch.device("cpu")#("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"): print("USING GPU")
        
    config = json.load(open(conf))
    exec(open(config["modelclass"]).read())
    model = load_model(config["model"], device=device)
    activation = {}
    list_layers = set_up_activations(model)

    som_size = config["som_size"]
    base_som_dir = config["base_som_dir"]
    base_spe = 'painters/base_spe'
    SOMs = {}
    SAE = {}
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
            oacts = acts.clone()
            acts = (acts-mm[layer]["min"])/(mm[layer]["max"]-mm[layer]["min"])
            
            ### --------- Start of the stting up the SOM --------- ### 
            
            
            
            if layer not in SOMs: 
                print("      *** creating", layer)
                
                encoding_dim = 3 * acts.size()[1]              
                SAE[layer] = SparseAutoencoder(acts.size()[1], encoding_dim, beta=1e-5, rho=5e-6).to(device)  
                                    
                
                
                #visualize_neuron_activity_all(encoded_activations, display_count=12, row_length=4)
                
                
               
# =============================================================================
#                 not working -> idea to find idx of all inactive neurons ... 
# =============================================================================
                #idx = np.where(np.all(encoded_activations, axis=0)<8e-4)[0]
                #idx_rand = np.random.choice(idx, 3, replace=False)
                #visualize_neuron_activity(layer, encoded_activations, idx_rand)
                
                # neuron_index= 3
                # check_neuron(acts.cpu().detach().numpy(), decoded_activations, neuron_index=neuron_index)
               
                
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
            
            print("   *** adding to SOM for",layer)
            change,count = SOMs[layer].add(acts.to(device))
            print("    train autoencoder")
            encoded_activations, \
            decoded_activations, \
            trained_autoencoder, \
            reconstruction_losses, \
            sparsity_penalties, \
            total_losses = train_SparseAE(SAE[layer],base_spe, layer,
                                               device,
                                               oacts.cpu().detach().numpy(), 
                                               oacts.size()[1]*3.0)   
            neuron_index= 3
            check_neuron(oacts.cpu().detach().numpy(), decoded_activations, neuron_index=neuron_index)
            print(f"      {count}/{len(acts)} elements resulted in a change of {change}")
            torch.save(SOMs[layer], base_som_dir+"/"+layer+".pt")
            # NaNs happen quickly, from first relu layer.
            # this is a trick... should be investigated why we get NaNs in the SOM
            if torch.isnan(SOMs[layer].somap).any(): 
                print ("*** NaN!")
                SOMs[layer].somap = torch.nan_to_num(SOMs[layer].somap, 0.0)   
        print("*** done ***")
        print ("*** integration of the trained sparce autoencoder and painter model")
        
