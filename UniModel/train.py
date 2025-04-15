import torch
from sklearn.metrics import r2_score

from model import UniModel
from dataset import UniDataset, load_dataset

VOCAB_SIZE=10000 # size of the vocab (number of words)
SEED = 42 # random seed 
BATCH_SIZE = 16 # size of the batches in training
EMB_SIZE = 256 # size of the embedding layer
LSTM_SIZE = 128 # size of the LSTM 
HIDDEN_SIZES = (64,32) # size of the hidden layers
DROPOUT = 0.3 # dropout rate
LEARNINGRATE = 0.0005 # learning rate
NEPOCHS = 200 # number of epochs (iteration of training)

# set the random seed so we always get the same results
torch.manual_seed(SEED)

device = "cpu"
if torch.cuda.is_available():
    print("GPU available")
    device = "cuda"
    
train_ds, val_ds, tdfmean, tdfstd = load_dataset(VOCAB_SIZE, split=True, SEED=SEED)

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=BATCH_SIZE)

# creating the model
model = UniModel(VOCAB_SIZE, EMB_SIZE, LSTM_SIZE, HIDDEN_SIZES, DROPOUT).to(device)
# optimizer for the training
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNINGRATE)
# loss function
criterion = torch.nn.MSELoss() # mean square error
# criterion = torch.nn.L1Loss() # mean average error

minloss = None
minep = 0
for epoch in range(1,NEPOCHS+1):
    model.train()
    runningloss,count = 0,0
    for texts,labels in train_loader:
        optimizer.zero_grad()
        texts = texts.to(device)
        labels = labels.to(device)
        out = model(texts) # calling the model
        loss = criterion(out,labels.reshape((-1, 1))) # computre de loss
        loss.backward() # compute gradient
        optimizer.step() # training step
        runningloss += loss # compute loss aggregation
        count+=1
    model.eval()
    # validation
    vcount,vrunningloss = 0,0
    with torch.no_grad(): 
      for texts,labels in val_loader:
          out = model(texts.to(device)) 
          loss = criterion(out.cpu(),labels.reshape((-1, 1)))
          vrunningloss += loss
          vcount+=1
    if minloss == None or vrunningloss/vcount < minloss:
        minloss = vrunningloss/vcount
        minep = epoch
        torch.save(model, f"model.pkl")
    print(f"{epoch}:: training loss={runningloss/count}, validation loss={vrunningloss/vcount}")
print(f"best model with loss {minloss} at epoch {minep}")
print("mean:", tdfmean, "std:", tdfstd)

bm = torch.load(f"model.pkl", weights_only=False)
preds = []
targets = []
with torch.no_grad(): 
    for texts,labels in val_loader:
        out = model(texts.to(device))
        preds = preds + out.reshape((-1)).detach().cpu().tolist()
        targets = targets + labels.detach().cpu().tolist()
r2 = r2_score(targets, preds)
print("R2 score:", r2)