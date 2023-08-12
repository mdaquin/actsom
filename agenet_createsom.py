import torch
import numpy as np
import os
import sys

import actsom

base_soms = {}
concepts = {}

# need to have acts in a directory and can be generic
# get all files and need an output dir...

# apply to painters..

act = torch.load("agenet/agenet_act_1")
for k in act:
    print(k,act[k][0].shape)
    sys.stdout.flush()
#    asom = actsom.ActSom(act[k], test=True)
#    asom.save("test_"+k+".pickle")

for k in act: base_soms[k] = None
for k in act:
    print("===",k,"===")
    if len(act[k][0].shape) == 3 and (act[k][0].shape[0]*act[k][0].shape[1]*act[k][0].shape[2]) > 100000:
        print("too large, skipping")
        continue
    sys.stdout.flush()
    acts_k = []
    count = 1
    while os.path.isfile("agenet/agenet_act_"+str(count)):
        print("   ",count)
        sys.stdout.flush()
        act = torch.load("agenet/agenet_act_"+str(count))
        acts_k.extend(act[k])
        count+=1
    asom = actsom.ActSom(acts_k)
    asom.save("agenet/soms/base_som_"+k+".pickle")
