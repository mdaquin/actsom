import torch.nn as nn

class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        
        self.linear = nn.Linear(self.n_features,self.n_features)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.n_features, 1)
        
    def forward(self, x):
        out = self.net(x)
        out = self.linear(out)
        out = self.relu(out)
        out = self.out(out)
        return out
