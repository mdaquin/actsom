import torch
import numpy as np
import os
import sys
import json

import actsom

if len(sys.argv) != 2:
    print("provide config file")
    sys.exit(1)

config = None

with open(sys.argv[1]) as f:
    config = json.load(f)
    
for li,layer in enumerate(config["layers"]):
    print("===",layer,"===")
    sys.stdout.flush()
    acts_k = []
    for f in config["act_files"]:
        print("   ",f)
        sys.stdout.flush()
        act = torch.load(f)
        acts_k.extend(act[layer])
    agg = None
    if config["aggregations"][li] == "mean": agg=torch.mean
    if config["aggregations"][li] == "flatten": agg=torch.flatten
    asom = actsom.ActSom(acts_k,somdim=tuple(config["som_dim"]),d2agg=agg)
    asom.save(config["outdir"]+"base_"+layer+".pickle")
