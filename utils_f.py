import torch
import matplotlib.pyplot as plt    
import numpy as np
import sys


def load_model(fn, device="cpu"):
    return torch.load(fn, map_location=device, weights_only=False)

def load_model_alignn(config,device='cpu'):
    from models.MATBENCH.alignn.models.alignn import ALIGNN, ALIGNNConfig
    from jarvis.db.jsonutils import loadjson
    import pprint
    from models.MATBENCH.alignn.config import TrainingConfig
    filename = config['model']
    config_model = loadjson(config['model_json'])

    print(pprint.pprint(config_model))
    config = TrainingConfig(**config_model)
    model = ALIGNN(config.model)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model



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
    
    


def check_neuron(layer,activation_data, decoded_activations, neuron_index):
    
    num_samples = activation_data.shape[0]

    if neuron_index >= activation_data.shape[1] or neuron_index >= decoded_activations.shape[1]:
        print(f"Error: Neuron index {neuron_index} is out of bounds.")
        return
    
    if activation_data.shape[1] == 1 :
        neuron_index = 1 
    
    original_neuron_data = activation_data[:, neuron_index].cpu().detach().numpy()
    decoded_neuron_data = decoded_activations[:, neuron_index].cpu().detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_samples), original_neuron_data, label=f'Original Neuron {neuron_index}', alpha=0.7, color='black')
    plt.plot(range(num_samples), decoded_neuron_data, label=f'Reconstructed Neuron {neuron_index} (after SAE)', alpha=0.3,linestyle='--',color='red')
    plt.title(f'{layer}')
    plt.xlabel('vector length')
    plt.ylabel('Activation Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_ICE(df,perturb_values,perturb_range,feature_index,layer,actdir):
    
    activation_data = torch.load(actdir+"/act_"+layer+".pt")
    decoded_activations = torch.load(actdir+"/decoded_"+layer+".pt")
    
    fig = plt.figure(figsize=(16, 12)) 
    ax1 = fig.add_subplot(2, 2, 1) 
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    for index, row in df.iterrows():
        original_predictions = np.array(row['prediction_original']).flatten()
        differences = [[] for _ in range(len(original_predictions))]
        differences_m = []

        for val in perturb_values:
            perturbation_col = f'perturbation_value_{val:.1f}'
            if perturbation_col in df.columns:
                perturbed_predictions = np.array(row[perturbation_col]).flatten()
                diff = original_predictions - perturbed_predictions
                diff_m = diff.mean()
                differences_m.append(diff_m)
                for i, d in enumerate(diff):
                    if i < len(original_predictions):
                        differences[i].append(d)                  
        for i in range(len(original_predictions)):
            ax1.plot(perturb_values, differences[i], '-', lw=0.5, alpha=0.7, color='blue')

    ax1.plot(perturb_values, differences_m, '--', lw=1.0,  color='black')
    ax1.set_ylabel(r'($P$-$P^{\prime}$)',fontsize=14)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.set_xlabel(f'Perturbation range that was applied on the neuron  {feature_index}',fontsize=12)

    ax1.grid(True)

    for index, row in df.iterrows():
        original_predictions = np.array(row['prediction_original']).flatten()
        differences = [[] for _ in range(len(original_predictions))]
        differences_m = []

        for val in perturb_values:
            perturbation_col = f'perturbation_value_{val:.1f}'
            if perturbation_col in df.columns:
                perturbed_predictions = np.array(row[perturbation_col]).flatten()
                diff = perturbed_predictions
                diff_m = diff.mean()
                differences_m.append(diff_m)
                for i, d in enumerate(diff):
                    if i < len(original_predictions):
                        differences[i].append(d)                  
        for i in range(len(original_predictions)):
            ax2.plot(perturb_values, differences[i], '-', lw=0.5, alpha=0.7, color='blue')

    ax2.plot(perturb_values, differences_m, '--', lw=1.0,  color='black')
    ax2.set_ylabel(r'$P^{\prime}$',fontsize=14)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(True)
    ax2.set_xlabel(f'Perturbation range that was applied on the neuron  {feature_index}',fontsize=12)
    
    
    num_samples = activation_data.shape[0]

    if feature_index >= activation_data.shape[1] or feature_index >= decoded_activations.shape[1]:
        print(f"Error: Neuron index {feature_index} is out of bounds.")
        return
    
    if activation_data.shape[1] == 1 :
        feature_index = 0 
    
    original_neuron_data = activation_data[:, feature_index].cpu().detach().numpy()
    decoded_neuron_data = decoded_activations[:, feature_index].cpu().detach().numpy()

    ax3.plot(range(num_samples), original_neuron_data, label=f'Original Neuron {feature_index}', alpha=0.7, color='black')
    ax3.plot(range(num_samples), decoded_neuron_data, label=f'Reconstructed Neuron {feature_index} (after SAE)',linestyle='--',color='red')
    ax3.set_title(f'{layer}')
    ax3.set_xlabel('vector length')
    ax3.set_ylabel('Activation Value')
    ax3.legend()
    plt.grid(True)
    
    
    
    
    
    
    
    
    plt.tight_layout() # Improves spacing between subplots
    plt.show()    

def get_module_by_name(model, name):
    parts = name.split('.')
    mod = model
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]  
        else:
            mod = getattr(mod, p)
    return mod

def set_module_by_name(model, name, new_module):
    parts = name.split('.')
    mod = model
    for p in parts[:-1]:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    last = parts[-1]
    if last.isdigit():
        mod[int(last)] = new_module
    else:
        setattr(mod, last, new_module)

