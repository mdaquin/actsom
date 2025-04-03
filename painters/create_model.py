import copy
import time
import torch
from torch.utils.data import Dataset, DataLoader
import json 
import torch.nn.functional as F
 
import torch.nn as nn
import torch 
    
class PainterDataset(Dataset):
    def __init__(self, dirname, range=None, device="cpu"):
        self.dirname=dirname
        self.list=[]
        self.device = device
        if range is None: range=range(1, 6)
        for i in range: self.list += json.load(open(f"{self.dirname}/dataset_{i}.json"))
        # TODO : create the tensor here to avoid the to(device) later

    def __len__(self): return len(self.list)

    def __getitem__(self, idx): return torch.tensor(self.list[idx]["I"], device=self.device), torch.tensor(self.list[idx]["O"], device=self.device)

class PainterModel(torch.nn.Module):
    def __init__(self,voc_size,emb_size,lstm_size,hidden_size, dor):
        super(PainterModel,self).__init__()
        self.dor = dor
        self.embedding = torch.nn.Embedding(num_embeddings=voc_size,embedding_dim=emb_size)
        self.lstm = torch.nn.LSTM(emb_size,lstm_size,bidirectional=True,batch_first=True)
        self.hidden = torch.nn.Linear(lstm_size*4,hidden_size)
        # self.relu = torch.nn.ReLU()
        self.out  = torch.nn.Linear(hidden_size, 1)

    def forward(self,x):
        x = self.embedding(x)
        x,_ = self.lstm(x)
        avg_pool = torch.mean(x,1)
        max_pool,_ = torch.max(x,1)
        out = torch.cat((avg_pool,max_pool),1)
        out = self.hidden(out)
        # out = self.relu(out)
        out = F.dropout(out, p=self.dor, training=self.training)
        out = self.out(out)
        out = torch.sigmoid(out)
        return out





# =============================================================================
# 
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
          
        return total_loss
    
    

#################3


class PainterModel_SPE(torch.nn.Module):
    def __init__(self,voc_size,emb_size,lstm_size,hidden_size, dor, encoding_dim, beta=1e-5, rho=5e-6):
        super(PainterModel_SPE,self).__init__()
        
        self.dor = dor
        self.embedding = torch.nn.Embedding(num_embeddings=voc_size,embedding_dim=emb_size)
        self.lstm = torch.nn.LSTM(emb_size,lstm_size,bidirectional=True,batch_first=True)
        
        self.spe = SparseAutoencoder(lstm_size*4, encoding_dim, beta, rho)
        
        #self.hidden = torch.nn.Linear(lstm_size*4,hidden_size)
        
        
        # self.relu = torch.nn.ReLU()
        self.out  = torch.nn.Linear(encoding_dim, 1)

    def forward(self,x):
        x = self.embedding(x)
        x,_ = self.lstm(x)
        avg_pool = torch.mean(x,1)
        max_pool,_ = torch.max(x,1)
        out = torch.cat((avg_pool,max_pool),1)
        
        #out = self.hidden(out)
        
        decoded, encoded = self.spe(out)
        
        spe_loss        = self.spe.compute_loss(out, decoded, encoded)
           
        # out = self.relu(out)
        out = F.dropout(encoded, p=self.dor, training=self.training)
        out = self.out(out)
        out = torch.sigmoid(out)
        return out, spe_loss


# =============================================================================
# 
# =============================================================================

def train():
    model.train()
    for x,y in train_loader:  # Iterate in batches over the training dataset.
        
        #out = model(x)
        #loss = criterion(out, y.to(torch.float32).view(len(y), -1))  # Compute the loss.
        #loss.backward()  # Derive gradients.
        out, spe_loss= model(x)
        critretion_loss = criterion(out, y.to(torch.float32).view(len(y), -1))        
        total_loss = spe_loss + critretion_loss
        
        total_loss.backward()
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model, loader, show=False, clear=False):
     model.eval()
     err = 0
     count=0
     for x,y in loader:  # Iterate in batches over the training/test dataset.
         #out = model(x)
         #out = (out>=0.5).to(torch.float32)
         out, spe_loss= model(x)
         critretion_loss = criterion(out, y.to(torch.float32).view(len(y), -1))        
         total_loss = spe_loss + critretion_loss
         
         out = (out>=0.5).to(torch.float32)
         # out = out.round()
         err += (out.T[0]-y).abs().sum()
         count += len(y)
     return 1-(err/count)

SEED = 43
DOR = 0.25
EPS = 10
BS = 32
ES = 128
LS = 128
MS = 12
LR = 0.001

torch.manual_seed(SEED)
device = "cpu"
if torch.cuda.is_available(): 
    print("RUNNING ON GPU")
    device = "cuda:0"

#traind = PainterDataset("painters/dataset", range(1,5), device)
#testd = PainterDataset("painters/dataset", range(5,6), device)

traind = PainterDataset("dataset", range(1,5), device)
testd = PainterDataset("dataset", range(5,6), device)

print("Train size:",len(traind),"Test size:", len(testd))

train_loader = DataLoader(traind, batch_size=BS, shuffle=True)
test_loader = DataLoader(testd, batch_size=1024, shuffle=False)

model=PainterModel_SPE(3843, ES, LS, MS, DOR,encoding_dim=100).to(device)
criterion = torch.nn.BCELoss() 
optimizer = torch.optim.Adam(model.parameters(),lr = LR)

ttt,tte = 0,0
best_test = None
for epoch in range(1, EPS+1):
    t1 = time.time()
    train()
    tt = round((time.time()-t1)*1000)
    ttt += tt
    t1 = time.time()
    train_acc = test(model, train_loader, show=True, clear=True)
    test_acc = test(model, test_loader, show=True)
    te = round((time.time()-t1)*1000)
    tte += te
    if best_test is None or test_acc > best_test:
        best_test = test_acc
        best_model = copy.deepcopy(model)
        best_epoch = epoch
    print(f'Epoch: {epoch:03d} ({tt:04d}/{te:04d}), Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f} (best: {best_test:.4f})')

print("Best acc on test", float(best_test),"at epoch",best_epoch)
print(f"Total time {round(ttt/1000):03d}s for training, {round(tte/1000):03d}s for testing")
print(f"Average time per epoch {round(ttt/EPS):04d}ms for training, {round(tte/EPS):04d}ms for testing")
#torch.save(best_model, "painters/model_spe.pt")
torch.save(best_model, "model_spe.pt")
print("saved model")