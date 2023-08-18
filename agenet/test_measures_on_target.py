import sys
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import entropy 
import seaborn as sns
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

sys.path.append("../")
from actsom import ActSom

layers=["net.layer1", "net.layer2", "net.layer3", "net.layer4", "net.avgpool", "relu"]

res = {
    "layers":layers,
    "entropy C0": [],
    "entropy C1": [],
    "entropy C2": [],    
    #"activation_entropy_0": [],
    #"activation_entropy_1": [],
    "max FM C0": [],
    "max FM C1": [],
    "max FM C2": [],        
    "distance C0": [],
    "distance C1": [],
    "distance C2": [],    
    #"activation_distance_0": [],
    #"activation_distance_1": [],
    "rel. entropy C0": [],
    "rel. entropy C1": [],
    "rel. entropy C2": [],    
    #"activation_rel_entropy_0": [],
    #"activation_rel_entropy_1": [],    
}

for layer in layers:

    bsom  = ActSom(f"soms/base_{layer}.pickle")
    t0som = ActSom(f"soms/target_[0.]_{layer}.pickle")
    t1som = ActSom(f"soms/target_[1.]_{layer}.pickle")
    t2som = ActSom(f"soms/target_[2.]_{layer}.pickle")
    
    res["entropy C0"].append(t0som.entropy())
    res["entropy C1"].append(t1som.entropy())
    res["entropy C2"].append(t2som.entropy())        
    #res["activation_entropy_0"].append(t0som.entropy(amap=True))
    #res["activation_entropy_1"].append(t1som.entropy(amap=True))

    res["max FM C0"].append(t0som.maxFM(bsom))
    res["max FM C1"].append(t1som.maxFM(bsom))
    res["max FM C2"].append(t2som.maxFM(bsom))        
    
    res["distance C0"].append(t0som.distance(bsom))
    res["distance C1"].append(t1som.distance(bsom))
    res["distance C2"].append(t2som.distance(bsom))    
    #res["activation_distance_0"].append(t0som.distance(bsom,amap=True))
    #res["activation_distance_1"].append(t1som.distance(bsom,amap=True))

    res["rel. entropy C0"].append(t0som.rel_entropy(bsom))
    res["rel. entropy C1"].append(t1som.rel_entropy(bsom))
    res["rel. entropy C2"].append(t2som.rel_entropy(bsom))    
    #res["activation_rel_entropy_0"].append(t0som.rel_entropy(bsom, amap=True))
    #res["activation_rel_entropy_1"].append(t1som.rel_entropy(bsom, amap=True))        
    
df = pd.DataFrame(res)
df = df.set_index("layers")

df["inv. entropy C0"] = 1/df["entropy C0"]
df["inv. entropy C1"] = 1/df["entropy C1"]
df["inv. entropy C2"] = 1/df["entropy C2"]

ranges={}
for k in df:
    ranges[k]={"min": df[k].min(), "max": df[k].max()}

rdf = pd.DataFrame(ranges)
print(rdf.T)
    
for k in df:
    df[k] = (df[k]-df[k].mean())/df[k].std()

df = df[["inv. entropy C0", "inv. entropy C1", "inv. entropy C2",
         "max FM C0", "max FM C1", "max FM C2",
         "distance C0", "distance C1", "distance C2",
         "rel. entropy C0", "rel. entropy C1", "rel. entropy C2"
         ]]
colours = ["darkblue", "blue", "lightblue", "indigo", "darkviolet", "violet", "darkgreen", "green", "lightgreen", "darkred", "red", "tomato"]
df.plot(color=colours)
plt.show()
    
#sns.heatmap(df.corr(), square=True)
#plt.show()
