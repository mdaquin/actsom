import sys
from minisom import MiniSom
import torch
import pickle
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from scipy.stats import entropy 
from sklearn.metrics.pairwise import cosine_distances

import matplotlib
matplotlib.use('TkAgg')

class ActSom:

    def __init__(self, X, somdim=(10,10),sigma=8,lr=0.1,d2agg=torch.flatten,normp=10.0,random_seed=None, test=False):
        if type(X) == bool:
            self.somdim = somdim
            self.d2agg = d2agg
            self.normp = normp
            self.sigma = sigma
            self.rs    = random_seed
            self.lr    = lr
            self.test  = test            
        elif type(X) == str:
            with open(X, 'rb') as infile:
                os = pickle.load(infile)
                self.som    = os.som
                self.grid   = os.grid
                self.amap   = os.amap
                self.somdim = os.somdim
                self.d2agg = os.d2agg
                self.normp = os.normp
                self.sigma = os.sigma
                self.rs    = os.rs
                self.lr    = os.lr
                self.test  = os.test
        else:
            self.somdim = somdim
            self.d2agg = d2agg
            self.normp = normp
            self.sigma = sigma
            self.rs    = random_seed
            self.lr    = lr
            self.test  = test
            iX=self.createInput(X)
            self.som = self.getSOM(iX)
            self.grid, self.amap = self.getGridMap(iX)
            
    def createInput(self,aX):
        nb  = len(aX)
        dim = aX[0].shape[0]
        if len(aX[0].shape) >= 2: dim = len(self.d2agg(aX[0],0))  
        X = torch.empty(size=(nb, dim))
        for i,x in enumerate(aX):
            if len(aX[0].shape) >= 2:
                v = self.d2agg(x,0)
                X[i] = v
            else: X[i] = x
        X=torch.nn.functional.normalize(X, p=self.normp, dim = 0).numpy()
        return X

    def getSOM(self,X):
        som = MiniSom(self.somdim[0], self.somdim[1], len(X[0]), sigma=self.sigma, learning_rate=self.lr, neighborhood_function='mexican_hat', random_seed=self.rs, activation_distance="cosine")
        if self.test: som.train(X, 10, verbose=True, use_epochs=False)        
        else: som.train(X, 1, verbose=True, use_epochs=True)
        return som

    def save(self, path):
        with open(path, 'wb') as outfile:
            pickle.dump(self, outfile)

    def getGridMap(self, X):
        grid = np.zeros((self.som._weights.shape[0],self.som._weights.shape[1]))
        activations = np.zeros((self.som._weights.shape[0],self.som._weights.shape[1]))
        for i,v in enumerate(X):
            x,y = self.som.winner(v)
            grid[x,y] += 1
            activations = activations + self.som._activation_map
            if (i!=0 and i%100==0):
                #print(i,end=" ")
                if self.test: break
        #print()                        
        #sys.stdout.flush()
        mactivations = activations/grid.sum()
        mactivations = mactivations/mactivations.sum()
        mactivations = 1-mactivations
        return grid,mactivations

    def populate(self, NX):
        res = ActSom(False, somdim=self.somdim, d2agg=self.d2agg, normp=self.normp, sigma=self.sigma, random_seed=self.rs, lr=self.lr, test=self.test)        
        res.som=self.som
        iX=res.createInput(NX)
        res.grid, res.amap = res.getGridMap(iX)
        return res

    def display(self,cmap="Greys",outfile=None):
        sns.heatmap(self.grid, cmap=cmap, cbar=False, norm=LogNorm())
        if outfile:
            plt.savefig(outfile)
            plt.close()
        else: plt.show()

    def entropy(self,amap=False):
        g = self.grid
        if amap: g = self.amap
        ng = (g / g.sum()).flatten()
        return entropy(ng,base=2)

    def rel_entropy(self,s,amap=False):
        g=self.grid
        bg=s.grid
        if amap:
            g=self.amap
            bg=s.amap
        nng = []
        nnbg = []
        ng = (g / g.sum()).flatten()
        nbg = (bg / bg.sum()).flatten()
        for i,v in enumerate(ng):
            if v != 0 and nbg[i]!=0 : 
                nng.append(v)
                nnbg.append(nbg[i])
        return entropy(nng, qk=nnbg, base=2)
            
    def distance(self, s, amap=False):
        g1=self.grid
        g2=s.grid
        if amap:
            g1=self.amap
            g2=s.amap
        ng1 = (g1 / g1.sum()).flatten()
        ng2 = (g2 / g2.sum()).flatten()
        return cosine_distances([ng1],[ng2]).flatten()[0]

    
    def precision(self, b):
        res = np.zeros(self.grid.shape)
        for i,l in enumerate(self.grid):
            for j,v in enumerate(l):
                res[i,j] = 0 if b.grid[i][j]==0 else v/b.grid[i][j]
        return res

    def recall(self):
        res = np.zeros(self.grid.shape)
        for i,l in enumerate(self.grid):
            for j,v in enumerate(l):
                res[i,j] = v/self.grid.sum()
        return res


    def fm(self, b):
        res = np.zeros(self.grid.shape)
        pr = self.precision(b)
        re = self.recall()
        for i,l in enumerate(pr):
            for j,p in enumerate(l):
                res[i,j] = 2*((p*re[i,j])/(p+re[i,j])) if p+re[i,j] != 0 else 0
        return res

    def maxFM(self, s, amap=False):
        return self.fm(s).max()

    
