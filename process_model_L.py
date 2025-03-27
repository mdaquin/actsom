import pandas as panda
import torch
import sys, os
import json
from ksom import SOM, cosine_distance, nb_gaussian, nb_linear, nb_ricker
import torch.nn as nn
import matplotlib.pyplot as plt    
from cmcrameri import cm
import numpy as np
from custom_functions import visualize_neuron_activity_all, check_neuron, plot_training_errors

# =============================================================================
# https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html
# D = d_model, F = dictionary_size
# e.g. if d_model = 12288 and dictionary_size = 49152
# then model_activations_D.shape = (12288,) and encoder_DF.weight.shape = (12288, 49152)
# =============================================================================


class SparseAutoencoder(nn.Module):
    
    '''    
    Args: 
        encoding_dim: number of hidden layers    
    '''
    
    def __init__(self, input_dim, encoding_dim, beta=0.01, rho=0.1):
        super(SparseAutoencoder, self).__init__()
       
        self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, input_dim), nn.Sigmoid())
        
        self.beta = beta
        self.rho = rho 

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded




    def compute_loss(self, x, decoded, encoded, eps = 1e-27):
    
        '''
        Computes total LOSS during the training SPE 
        
        Theory: 
            sparcity penaly : beta * sum (KL(rho|| rho_hat_j))_j^s, where s is the number of hidden layers (encoding_dim)
            
            Kullback-Leibler (KL) Divergence: measures the difference between the desired sparsity (rho) 
            and the actual average activation (rho_hat); A higher value means the neuron is deviating more from the target sparsity level. 
            KL(ρ∣∣ρ^​j​)=ρlog(​ρ/ρ^​j)​+(1−ρ)log[(1−ρ)/(1-ρ^​j)]​
        
        Args:
            beta: sparsity loss coefficient or weitgh of sparcite penalty 
            rho : the desired sparsity
            rho_hat : the actual average activation 
            eps: to avoid devision by zero     
        
        '''
        rho_hat = torch.mean(encoded, dim=0)
        rho_hat = torch.clamp(rho_hat, min=eps, max=1 - eps)
        KL_div = self.rho * torch.log((self.rho / rho_hat)) + (1 - self.rho) * torch.log(((1 - self.rho) / (1 - rho_hat))) 
        sparcity_penalty = self.beta * torch.sum(KL_div)
        
        #print("rho_hat =  ",rho_hat)
        #print ("Kullback-Leibler (KL) Divergence: ", torch.mean(KL_div).detach().numpy())
    
        reconstruction_loss = nn.MSELoss()(x, decoded)    
        total_loss = reconstruction_loss + sparcity_penalty 
          
        return total_loss,  reconstruction_loss.item(), sparcity_penalty.item() 
        



def train_SparseAE(layer, device, activations, encoding_dim, beta=0.1, rho=5e-4, epochs=1000, learning_rate=0.001):
    path = base_spe+"/loss_res/"
    for beta in [1e-4,1e-1,1e0,1e1,1e2]:
        for rho in [5e-6,5e-4,5e-2]:
            
    
            reconstruction_losses = []
            sparsity_penalties = []
            total_losses = []
            
            maxEr = np.inf 
            
            input_dim = activations.shape[1]
            activation_transformed = torch.tensor(activations, dtype=torch.float32).to(device)
            autoencoder = SparseAutoencoder(input_dim, encoding_dim, beta=beta, rho=rho).to(device)
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                decoded, encoded = autoencoder(activation_transformed)
                total_loss, recon_loss_val, sparsity_val = autoencoder.compute_loss(activation_transformed, decoded, encoded)
                if total_loss < maxEr:
                        torch.save(autoencoder,base_spe+"/"+layer+".pt")
                total_loss.backward()
                optimizer.step()
        
                reconstruction_losses.append(recon_loss_val)
                sparsity_penalties.append(sparsity_val)
                total_losses.append(total_loss.item())
                print(f'Sparse AE Epoch {epoch+1}, Total Loss: {total_loss.item():.4f}, Recon Loss: {recon_loss_val:.4f}, Sparsity: {sparsity_val:.4f}')
        
            encoded_activations = autoencoder.encoder(activation_transformed).detach().cpu().numpy()
            decoded_activations = autoencoder.decoder(autoencoder.encoder(activation_transformed)).detach().cpu().numpy() 
        
            plot_training_errors(path,rho,layer, beta, reconstruction_losses, sparsity_penalties, total_losses)

    return encoded_activations, decoded_activations, autoencoder, reconstruction_losses, sparsity_penalties, total_losses









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
            acts = (acts-mm[layer]["min"])/(mm[layer]["max"]-mm[layer]["min"])
            
            ### --------- Start of the stting up the SOM --------- ### 
            
            
            
            if layer not in SOMs: 
                print("      *** creating", layer)
                
                encoding_dim = 3 * acts.size()[1]                
                                    
                encoded_activations, \
                decoded_activations, \
                trained_autoencoder, \
                reconstruction_losses, \
                sparsity_penalties, \
                total_losses = train_SparseAE(layer,
                                              device,
                                              acts.cpu().detach().numpy(), 
                                              encoding_dim)   

                
                
                print(f"Encoded Activations shape for layer {layer}:", encoded_activations.shape)
                print(f"Decoded Activations shape for layer {layer}:", decoded_activations.shape)
                
                #visualize_neuron_activity_all(encoded_activations, display_count=12, row_length=4)
                
                
               
# =============================================================================
#                 not working -> idea to find idx of all inactive neurons ... 
# =============================================================================
                #idx = np.where(np.all(encoded_activations, axis=0)<8e-4)[0]
                #idx_rand = np.random.choice(idx, 3, replace=False)
                #visualize_neuron_activity(layer, encoded_activations, idx_rand)
                
                neuron_index= 3
                #check_neuron(acts.cpu().detach().numpy(), decoded_activations, neuron_index=neuron_index)
               
                
                
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
            print(f"      {count}/{len(acts)} elements resulted in a change of {change}")
            torch.save(SOMs[layer], base_som_dir+"/"+layer+".pt")
            # NaNs happen quickly, from first relu layer.
            # this is a trick... should be investigated why we get NaNs in the SOM
            if torch.isnan(SOMs[layer].somap).any(): 
                print ("*** NaN!")
                SOMs[layer].somap = torch.nan_to_num(SOMs[layer].somap, 0.0)   
        print("*** done ***")
