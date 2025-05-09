from random import shuffle
import torch, sys, json
import importlib as imp
import utils_f as u
import SPE.sparce_autoencoder as SAE
import pickle
import matgl


def get_activations_megnet(t):
        if type(t) == tuple and len(t) > 1: t = t[0] # only the first one for now...
        if type(t) == tuple and len(t) > 1: t = t[0] # can be a tuple in a tuple
        if len(t.shape) == 1: return t.detach()
        if len(t.shape) == 2:
            if t.shape[0] == 1: return t[0].detach()
            else: return torch.mean(t, dim=1) # could be other aggregation methods
        if len(t.shape) == 3:
            if t.shape[0] == 1 and t.shape[1] == 1: return t[0][0].detach()
            else: return torch.mean(torch.mean(t, dim=2), dim=1).detach() # randomly, I don't think this happens
        # print("!!!!!!!!!!!!", len(t.shape))
        return None

def trainSAE(SAEs, optimizers, layer, acts, SAEevs):
    if layer not in SAEs:
        factor = config["basefactor"] 
        while acts.size()[1]*acts.size()[1]*factor*4 > config["maxsize"]:
            factor -= 1
        if factor < 1: 
            SAEs[layer] = None   
            print(f"{layer} too large, skipping")
        else:    
            encoding_dim = int(acts.size()[1] * factor)
            print(f"Encoding dim for {layer} = {encoding_dim} (factor={factor})")
            SAEs[layer] = SAE.SparseAutoencoder(acts.size()[1], 
                                                encoding_dim, 
                                                beta=config["beta"], 
                                                rho=config["rho"]).to(device) 
            SAEev[layer] = {"rec": 0, "sparse": 0, "loss": 0, "min": None}
            optimizers[layer] = torch.optim.Adam(SAEs[layer].parameters(), 
                                                 lr=config["learningrate"])
            SAEs[layer].train()
    if SAEs[layer] is None: return
    optimizers[layer].zero_grad()
    decoded, encoded = SAEs[layer](acts)
    total_loss, recon_loss_val, sparsity_val = SAEs[layer].compute_loss(acts, decoded, encoded)
    total_loss.backward()
    optimizers[layer].step()
    SAEev[layer]["rec"] += recon_loss_val
    SAEev[layer]["sparse"] += sparsity_val
    SAEev[layer]["loss"] += total_loss
    if SAEev[layer]["min"] is None or total_loss < SAEev[layer]["min"]:
        SAEev[layer]["min"] = total_loss
        torch.save(SAEs[layer], f"{config['saedir']}/{layer}.pkl")
 
if __name__ == "__main__":
     if len(sys.argv) != 2:
         print("Usage: python process_model.py <config_file.json>")
         sys.exit(1)
     
     config = json.load(open(sys.argv[1]))
     #config = json.load(open("configurations/config_painters.json"))
     torch.manual_seed(config["seed"])
     som_size = config["som_size"]
     base_som_dir = config["somdir"]
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     runcpu = "runcpu" in config and config["runcpu"]
     if runcpu: device = torch.device("cpu")
     if device == torch.device("cuda"): print("USING GPU")

     print("Loading model...")   
     with torch.no_grad():
        model = matgl.load_model("megnet/model")

     print("Loading dataset...")
     with open("megnet/data/mp.2018.6.1_structures.pkl", "rb") as f:
        structures = pickle.load(f)
     with open("megnet/data/mp.2018.6.1_mp_ids.pkl", "rb") as f:
        mp_ids = pickle.load(f)
 
     print("Setting up activation hooks...")
     u.activation = {}
     list_layers = u.set_up_activations(model)
     print(list_layers)
      
     print("Training SAEs...")
     SAEs = {}
     SAEev = {}
     optimizers = {}
     for ep in range(1, config["nepochs_sae"]+1):
         count = 0
         sev = 0
         batch = {}
         shuffle(structures)
         # for X,y in data_loader:
         for i, struct in enumerate(structures):
             u.activation = {}
             pred = model.predict_structure(struct)        
             for layer in u.activation:
                acts = get_activations_megnet(u.activation[layer])
                if acts is None or len(acts) <= 1 : continue
                # print(layer, "::", acts.shape)
             # create a batch of batch_size
                if layer not in batch: batch[layer] = []
                else: 
                    if len(batch[layer]) != 0 and len(acts) != len(batch[layer][0]):
                        # print("Bad layer: ", layer)
                        batch[layer][0] = []
                    else: 
                        batch[layer].append(acts)
                if len(batch[layer]) >= config["batchsize"]:
                    #print("layer batch", layer) 
                    # print(len(batch[layer]))
                    # trainSOM(torch.stack(batch[layer], dim=0), layer, SOMs, mm, SOMevs)
                    # print(".", end="")
                    trainSAE(SAEs, optimizers, layer, torch.stack(batch[layer], dim=0), SAEev)
                    batch[layer] = []
             count += 1
             if i%100==0: print(".", end="")
         print()        

         for layer in SAEev:
             SAEev[layer]["rec"] /= count
             SAEev[layer]["sparse"] /= count
             SAEev[layer]["loss"] /= count
             print(f"   {layer}:: rec={SAEev[layer]['rec']}, sparse={SAEev[layer]['sparse']}, loss={SAEev[layer]['loss']}")
 