import sys
import pandas as pd 
import matplotlib.pyplot as plt
import glob

import matplotlib
matplotlib.use('TkAgg')

sys.path.append("../")
from actsom import ActSom

layers=["net.layer1", "net.layer2", "net.layer3", "net.layer4", "net.avgpool", "relu"]

concept_att= [
    "gender",
    "ethnicity"]

for c in concept_att:
    gdf = pd.DataFrame()
    for l in layers:
        if l=="hidden": continue
        data = {"concept": [], "distance": [], "rel. entropy": [], "len":[]}
        bsom  = ActSom(f"soms/base_{l}.pickle")
        lfiles = glob.glob(f'soms/concept_{c}__*__{l}.pickle')        
        for f in lfiles:
            v=f[:f.rindex("__")]
            v=v[v.rindex("__")+2:]
            if ":" in v: v=v[v.index(":")+1:]
            cvsom = ActSom(f)
            data["concept"].append(v)
            data["distance"].append(cvsom.distance(bsom))
            data["rel. entropy"].append(cvsom.rel_entropy(bsom))
            data["len"].append(cvsom.grid.sum())
        df = pd.DataFrame(data)
        df = df.set_index("concept").sort_values("rel. entropy")
        gdf[l] = df["rel. entropy"]
        
    gdf.sort_values("relu").plot(kind="bar", title=f"{c}")
    plt.show()
