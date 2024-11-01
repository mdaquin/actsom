import torch 
import torch.nn.functional as F

class PainterModel(torch.nn.Module):
    def __init__(self,voc_size,emb_size,lstm_size,hidden_size, dor):
        super(PainterModel,self).__init__()
        self.dor = dor
        self.embedding = torch.nn.Embedding(num_embeddings=voc_size,embedding_dim=emb_size)
        self.lstm = torch.nn.LSTM(emb_size,lstm_size,bidirectional=True,batch_first=True)
        self.hidden = torch.nn.Linear(lstm_size*4,hidden_size)
        self.relu = torch.nn.ReLU()
        self.out  = torch.nn.Linear(hidden_size, 1)

    def forward(self,x):
        x = self.embedding(x)
        x,_ = self.lstm(x)
        avg_pool = torch.mean(x,1)
        max_pool,_ = torch.max(x,1)
        out = torch.cat((avg_pool,max_pool),1)
        out = self.hidden(out)
        out = self.relu(out)
        out = F.dropout(out, p=self.dor, training=self.training)
        out = self.out(out)
        out = torch.sigmoid(out)
        return out