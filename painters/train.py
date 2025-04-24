import copy, time, torch, json
from model import PainterModel
from dataset import load_dataset

def train():
    losssum, count = 0, 0
    model.train()
    for x,y in train_loader:  # Iterate in batches over the training dataset.        
        optimizer.zero_grad()  # Clear gradients.
        x = x.to(device) 
        y = y.to(device)
        out = model(x)
        loss = criterion(out, y.to(torch.float32).view(len(y), -1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        losssum = losssum+loss
        count += 1
    return losssum/count

def test(model, loader, show=False, clear=False):
     model.eval()
     err = 0
     count=0
     with torch.no_grad():  # No need to track history in test mode
      for x,y in loader:  # Iterate in batches over the training/test dataset.
         x = x.to(device) 
         y = y.to(device)
         out = model(x)
         out = (out>=0.5).to(torch.float32)
         # out = out.round()
         err += (out.T[0]-y).abs().sum()
         count += len(y)
     return 1-(err/count)

SEED = 43
DOR = 0.5 # dropout rate
EPS = 20 # number of epochs
BS = 32 # batch size
ES = 256 # embedding size
LS = 256 # lstm size
MS = 24 # hidden size
LR = 0.0005 # learning rate

torch.manual_seed(SEED)
device = "cpu"
if torch.cuda.is_available(): 
    print("RUNNING ON GPU")
    device = "cuda:0"

traind, testd = load_dataset(3000, split=True, SEED=SEED)

print("Train size:",len(traind),"Test size:", len(testd))

train_loader = torch.utils.data.DataLoader(traind, batch_size=BS, shuffle=True)
test_loader = torch.utils.data.DataLoader(testd, batch_size=256, shuffle=False)

model=PainterModel(3000, ES, LS, MS, DOR).to(device)
criterion = torch.nn.BCELoss() 
optimizer = torch.optim.Adam(model.parameters(),lr = LR)

ttt,tte = 0,0
best_test = None
for epoch in range(1, EPS+1):
    t1 = time.time()
    loss = train()
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
    print(f'Epoch: {epoch:03d} ({tt:04d}/{te:04d}), Train loss: {loss:.4f}, Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f} (best: {best_test:.4f})')

print("Best acc on test", float(best_test),"at epoch",best_epoch)
print(f"Total time {round(ttt/1000):03d}s for training, {round(tte/1000):03d}s for testing")
print(f"Average time per epoch {round(ttt/EPS):04d}ms for training, {round(tte/EPS):04d}ms for testing")
torch.save(best_model, "painters/model.pt")
print("saved model")
