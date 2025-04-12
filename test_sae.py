import os
import sys
import torch 
import utils as u
import json
import pandas as pd
import torch.nn.functional as F


class PainterModelSPE(torch.nn.Module):
    def __init__(self, embedding, lstm, hidden, out, sparse_autoencoder):
        super(PainterModelSPE,self).__init__()
        self.embedding = embedding
        self.lstm = lstm
        self.hidden = hidden
        self.sparse_autoencoder = sparse_autoencoder
        
        self.out  = out
        
    def forward(self,x):
        x = self.embedding(x)
        x,_ = self.lstm(x)
        avg_pool = torch.mean(x,1)
        max_pool,_ = torch.max(x,1)
        out = torch.cat((avg_pool,max_pool),1)
        
        out = self.hidden(out)
        # out = self.relu(out)
        
        decoded, encoded = self.sparse_autoencoder(out)

        # out = F.dropout(decoded, p=self.dor, training=self.training) # this needs to go...
        
        out = self.out(decoded)
        out = torch.sigmoid(out)
        return out


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

if len(sys.argv) != 2: 
    print("Please provide a config file.")
    sys.exit(-1)

config = json.load(open(sys.argv[1]))

print("*** loading model")
# exec(open(config["modelclass"]).read())
base_model=u.load_model(config["model"])

sae = config["base_sae_dir"]+"/hidden.pt"
sae = torch.load(sae, weights_only=False).to("cpu")
sae.eval()

print("Average weights of encoding layer: ", torch.mean(sae.encoder[0].weight).item())

sae_model = PainterModelSPE(base_model.embedding, base_model.lstm, base_model.hidden, base_model.out, sae)
sae_model.eval()

sae_model.embedding.load_state_dict(base_model.embedding.state_dict())
sae_model.lstm.load_state_dict(base_model.lstm.state_dict())
sae_model.hidden.load_state_dict(base_model.hidden.state_dict())

print("Average weights of embedding layer om: ", torch.mean(base_model.embedding.weight).item())
print("Average weights of hidden layer om: ", torch.mean(base_model.hidden.weight).item())

print("Average weights of embedding layer sm: ", torch.mean(sae_model.embedding.weight).item())
print("Average weights of hidden layer sm: ", torch.mean(sae_model.hidden.weight).item())

# create the SAE comb...

def eval(t,p):
    err = 0
    for i,pv  in enumerate(p):
        if pv >= 0.5 and t[i] == 0: err+=1
        elif pv < 0.5 and t[i] == 1: err+=1
    return 1-(err/len(t))

results = {"target": [], "pred": []}
dataset = u.KSDataset(config["dataset_dir"], return_c=True)
for i in range(len(dataset)):
    print("   *** file", i)
    IS,OS,CS, = dataset[i] # add the concetps
    if IS.to(int).equal(IS): IS = IS.to(int)
    target = list(OS.detach().numpy())
    PS = base_model(IS)
    pred = list(PS.detach().numpy().flatten())
    print("Précision: ", eval(target, pred))
    PS2 = sae_model(IS)
    pred2 = list(PS2.detach().numpy().flatten())
    print("Précision post SAE: ", eval(target, pred2))
    PS3 = sae_model(IS)
    pred3 = list(PS3.detach().numpy().flatten())
    print("Précision post SAE: ", eval(target, pred3))
