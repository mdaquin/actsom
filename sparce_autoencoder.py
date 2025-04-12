import torch.nn as nn
import torch 
import numpy as np 

class SparseAutoencoder(nn.Module):
    
    '''    
    Args: 
        encoding_dim: number of hidden layers    
    '''
    
    def __init__(self, input_dim, encoding_dim, beta=0.01, rho=0.01):
        super(SparseAutoencoder, self).__init__()
       
        self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, input_dim)) # MDA: changed here: non normalised acts can be negative 
        
        self.beta = beta # should be dependent on range of values in input dim
        self.rho = rho # should be dependent on encoding dim

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
    
    
def train_SparseAE(autoencoder, base_spe,layer, device, activations, encoding_dim, beta=1e-3, rho=5e-6, epochs=1000, learning_rate=0.001):
    
    reconstruction_losses = []
    sparsity_penalties = []
    total_losses = []
    
    maxEr = np.inf 
    
    activation_transformed = torch.tensor(activations, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        decoded, encoded = autoencoder(activation_transformed)
        total_loss, recon_loss_val, sparsity_val = autoencoder.compute_loss(activation_transformed, decoded, encoded)
        if total_loss < maxEr: torch.save(autoencoder,base_spe+"/"+layer+".pt")
        total_loss.backward()
        optimizer.step()

        reconstruction_losses.append(recon_loss_val)
        sparsity_penalties.append(sparsity_val)
        total_losses.append(total_loss.item())
    print(f'Sparse AE Epoch {epoch+1}, Total Loss: {total_loss.item():.4f}, Recon Loss: {recon_loss_val:.4f}, Sparsity: {sparsity_val:.4f}')

    with torch.no_grad():
        encoded_activations = autoencoder.encoder(activation_transformed).detach().cpu().numpy()
        decoded_activations = autoencoder.decoder(autoencoder.encoder(activation_transformed)).detach().cpu().numpy() 
    
            #     path = base_spe+"/loss_res/"

            # plot_training_errors(path,rho,layer, beta, reconstruction_losses, sparsity_penalties, total_losses)
            
    return encoded_activations, decoded_activations, autoencoder, reconstruction_losses, sparsity_penalties, total_losses
    