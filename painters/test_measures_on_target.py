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

layers=["embedding", "lstm", "hidden", "relu"]

res = {
    "layers":layers,
    "entropy_0": [],
    "entropy_1": [],
    "activation_entropy_0": [],
    "activation_entropy_1": [],
    "distance_0": [],
    "distance_1": [],
    "activation_distance_0": [],
    "activation_distance_1": [],
    "rel_entropy_0": [],
    "rel_entropy_1": [],
    "activation_rel_entropy_0": [],
    "activation_rel_entropy_1": [],    
}

for layer in layers:

    bsom  = ActSom(f"soms/base_{layer}.pickle")
    t0som = ActSom(f"soms/target_0.0_{layer}.pickle")
    t1som = ActSom(f"soms/target_1.0_{layer}.pickle")
    
    res["entropy_0"].append(t0som.entropy())
    res["entropy_1"].append(t1som.entropy())    
    res["activation_entropy_0"].append(t0som.entropy(amap=True))
    res["activation_entropy_1"].append(t1som.entropy(amap=True))

    res["distance_0"].append(t0som.distance(bsom))
    res["distance_1"].append(t1som.distance(bsom))
    res["activation_distance_0"].append(t0som.distance(bsom,amap=True))
    res["activation_distance_1"].append(t1som.distance(bsom,amap=True))

    res["rel_entropy_0"].append(t0som.rel_entropy(bsom))
    res["rel_entropy_1"].append(t1som.rel_entropy(bsom))

    res["activation_rel_entropy_0"].append(t0som.rel_entropy(bsom, amap=True))
    res["activation_rel_entropy_1"].append(t1som.rel_entropy(bsom, amap=True))        

ranges={}
for k in res:
    if k != "layers":
        ranges[k]={"min": np.array(res[k]).min(), "max": np.array(res[k]).max()}
    
rdf = pd.DataFrame(ranges)
print(rdf.T)
    
df = pd.DataFrame(res)
df = df.set_index("layers")
df["entropy"] = 1/((df.entropy_0+df.entropy_1)/2)
df["inv. entropy"] = (df.entropy-df.entropy.mean())/df.entropy.std()
df["act. entropy"] = 1/((df.activation_entropy_0+df.activation_entropy_1)/2)
df["inv. act. entropy"] = (df["act. entropy"]-df["act. entropy"].mean())/df["act. entropy"].std()
df["distance"] = (df.distance_0+df.distance_1)/2
df["distance"] = (df.distance-df.distance.mean())/df.distance.std()
df["act. distance"] = (df.activation_distance_0+df.activation_distance_1)/2
df["act. distance"] = (df["act. distance"]-df["act. distance"].mean())/df["act. distance"].std()
df["rel. entropy"] = (df.rel_entropy_0+df.rel_entropy_1)/2
df["rel. entropy"] = (df["rel. entropy"]-df["rel. entropy"].mean())/df["rel. entropy"].std()
df["act. rel. entropy"] = (df.activation_rel_entropy_0+df.activation_rel_entropy_1)/2
df["act. rel. entropy"] = (df["act. rel. entropy"]-df["act. rel. entropy"].mean())/df["act. rel. entropy"].std()

vdf = df[["inv. entropy", "inv. act. entropy", "distance", "act. distance", "rel. entropy", "act. rel. entropy"]]
print(vdf.T)
vdf.plot()
plt.show()


    

sns.heatmap(vdf.corr(), square=True)
plt.show()
