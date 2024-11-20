import copy
import time
import torch
from torch.utils.data import Dataset, DataLoader
import json 
import torch.nn.functional as F

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


def train():
    model.train()
    for x,y in train_loader:  # Iterate in batches over the training dataset.
        out = model(x)
        loss = criterion(out, y.to(torch.float32).view(len(y), -1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model, loader, show=False, clear=False):
     model.eval()
     err = 0
     count=0
     for x,y in loader:  # Iterate in batches over the training/test dataset.
         out = model(x)
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
traind = PainterDataset("painters/dataset", range(1,5), device)
testd = PainterDataset("painters/dataset", range(5,6), device)
print("Train size:",len(traind),"Test size:", len(testd))

train_loader = DataLoader(traind, batch_size=BS, shuffle=True)
test_loader = DataLoader(testd, batch_size=1024, shuffle=False)

model=PainterModel(3843, ES, LS, MS, DOR).to(device)
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
torch.save(best_model, "painters/model.pt")
print("saved model")