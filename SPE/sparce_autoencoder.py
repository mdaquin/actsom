import torch.nn as nn
import torch 
class SparseAutoencoder(nn.Module):
    
    '''    
    Args: 
        encoding_dim: number of hidden layers    
    '''
    
    def __init__(self, input_dim, encoding_dim, beta=0.01, rho=0.1):
        super(SparseAutoencoder, self).__init__()
       
        self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.Sigmoid())
        # self.decoder = nn.Sequential(nn.Linear(encoding_dim, input_dim), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, input_dim))

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
    

class SPE_Integrate(torch.nn.Module):
    def __init__(self,
                 base_model,
                 spe_model, 
                 feature_index
                 ):
        
        super(SPE_Integrate, self).__init__()
        self.base_model = base_model 
        self.spe_model = spe_model
        self.feature_index = feature_index
        self.val = 1.0  
    def set_val(self, val):
        self.val = float(val)
           
    def forward(self, *args, **kwargs):
            out = self.base_model(*args, **kwargs) 
            #print("### - >> we are here!")
            if isinstance(out, (tuple, list)):
                selected = out[self.output_index] if hasattr(self, "output_index") else out[0]
            else:
                selected = out
            
            
            #print("### - >> Selected!", selected.shape)
            #selected = out[self.feature_index]
            encoded =  self.spe_model.encoder(selected)
            
            #print("### - >> After encoding!", encoded.shape)
            
            encoded_ice = encoded.clone()
            #print("### size shape: ",out.shape)
            # specifications of CrabNet : 
            if  encoded_ice.dim() > 2 : 
                
                # for CrabNet : [data_len, out_dim, size of hidden layer]
                encoded_ice[:,:,self.feature_index] *=  torch.tensor(self.val, dtype=torch.float32) 
                
                # for ParAIsite: [ node_feat, edge_feat, state_feat] 
                #encoded_ice[self.feature_index,:,:] *=  torch.tensor(self.val, dtype=torch.float32)
                
            else:
                encoded_ice[:,self.feature_index] *=  torch.tensor(self.val, dtype=torch.float32)   
                
            
            #print("### - >> Before decoding!", encoded.shape)      
            decoded_ice = self.spe_model.decoder(encoded_ice) 
            #print("### - >> After decoding!", decoded_ice.shape)         
            return decoded_ice       
        

'''
    def forward(self, *args, **kwargs):  
        out = self.base_model(*args, **kwargs) 

        if not isinstance(out, (tuple, list)):
            out = (out,)
        out = list(out)  
        
        #print("out:", [o.shape if isinstance(o, torch.Tensor) else type(o) for o in out])

        selected = out[self.feature_index]

        encoded = self.spe_model.encoder(selected)
        encoded_ice = encoded.clone()
        encoded_ice[:, self.feature_index] *= torch.tensor(self.val, dtype=torch.float32, device=encoded.device)
        #encoded_ice[:, self.feature_index] = encoded_ice[:, self.feature_index] * encoded_ice[:, self.feature_index] ** torch.tensor(self.val, dtype=torch.float32, device=encoded.device)
        decoded_ice = self.spe_model.decoder(encoded_ice)

        out[self.feature_index] = decoded_ice

        if len(out) == 1:
            return decoded_ice
        else:
            return tuple(out)    
'''        