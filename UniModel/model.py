import torch 

class UniModel(torch.nn.Module):
    def __init__(self,
                 voc_size,  # vocabulary size
                 emb_size,  # embedding size
                 lstm_size, # lstm size
                 hidden_sizes, # size of hidden layers 
                 dor): # dropout rate
        super(UniModel,self).__init__()
        self.dor = dor
        self.embedding = torch.nn.Embedding(num_embeddings=voc_size,embedding_dim=emb_size)
        self.lstm = torch.nn.LSTM(emb_size,lstm_size,bidirectional=True,batch_first=True)
        self.hidden=[]
        prev = lstm_size*4
        for hs in hidden_sizes:
            self.hidden.append(torch.nn.Linear(prev,hs))
            prev = hs
        self.hidden = torch.nn.Sequential(*self.hidden)
        self.out  = torch.nn.Linear(prev, 1)

    def forward(self,x):
        x = self.embedding(x)
        x,_ = self.lstm(x)
        avg_pool = torch.mean(x,1)
        max_pool,_ = torch.max(x,1)
        out = torch.cat((avg_pool,max_pool),1)
        for hi in self.hidden:
            out = hi(out)
            out = torch.relu(out)
        out = torch.nn.functional.dropout(out, p=self.dor, training=self.training)
        out = self.out(out)
        # out = torch.sigmoid(out)
        return out
