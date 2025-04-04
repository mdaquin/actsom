#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPE and Painter model combo
"""


import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json 


device = "cpu"
SEED = 42        
torch.manual_seed(SEED)


def test_models (model): 
    
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
        
        

        testd = PainterDataset("painters/dataset", range(5,6), device)
        test_loader = DataLoader(testd, batch_size=1024, shuffle=False)
        model.eval()
        err = 0
        count=0
        for x,y in test_loader:  # Iterate in batches over the training/test dataset.
             out = model(x)
             out = (out>=0.5).to(torch.float32)
         # out = out.round()
             err += (out.T[0]-y).abs().sum()
             count += len(y)
        print("prediction_shape:", out.size(),"presicion:",1-(err/count))     
        return 1-(err/count)




class PainterModelSPE(torch.nn.Module):
    def __init__(self,voc_size,emb_size,lstm_size,hidden_size, dor,sparse_autoencoder):
        super(PainterModelSPE,self).__init__()
        self.dor = dor
        self.embedding = torch.nn.Embedding(num_embeddings=voc_size,embedding_dim=emb_size)
        self.lstm = torch.nn.LSTM(emb_size,lstm_size,bidirectional=True,batch_first=True)
        self.hidden = torch.nn.Linear(lstm_size*4,hidden_size)
        self.sparse_autoencoder = sparse_autoencoder
        
        #self.out  = torch.nn.Linear(hidden_size, 1) # before 
        
        
        
        encoding_dim = self.sparse_autoencoder.encoder[0].out_features
        self.out  = torch.nn.Linear(encoding_dim, 1) # now 
        
    def forward(self,x):
        x = self.embedding(x)
        x,_ = self.lstm(x)
        avg_pool = torch.mean(x,1)
        max_pool,_ = torch.max(x,1)
        out = torch.cat((avg_pool,max_pool),1)
        
        out = self.hidden(out)
        # out = self.relu(out)
        
        _, encoded = self.sparse_autoencoder(out)

        out = F.dropout(encoded, p=self.dor, training=self.training)
        
        out = self.out(out)
        out = torch.sigmoid(out)
        return out

        
## Load pre-trained model Painter ####

conf = "config_painters.json"
config = json.load(open(conf))
exec(open(config["modelclass"]).read())
model_original = torch.load(config["model"],map_location=device, weights_only=False)

## Load SPE 

spe_model = torch.load('painters/base_spe/hidden.pt',map_location=device, weights_only=False)


new_painter_model = PainterModelSPE(voc_size=3843, emb_size=128, lstm_size=128, hidden_size=12, dor=0.3,sparse_autoencoder=spe_model)

new_painter_model.embedding.load_state_dict(model_original.embedding.state_dict())
new_painter_model.lstm.load_state_dict(model_original.lstm.state_dict())
new_painter_model.hidden.load_state_dict(model_original.hidden.state_dict())

new_painter_model.sparse_autoencoder.encoder.load_state_dict(spe_model.encoder.state_dict())
new_painter_model.sparse_autoencoder.decoder.load_state_dict(spe_model.decoder.state_dict())


# Check predictions! 

test_models (new_painter_model)
test_models (model_original)


