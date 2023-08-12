import sys
import json

import torch

from actsom import ActSom

if len(sys.argv) != 2:
    print("provide config file")
    sys.exit(1)

config = None

with open(sys.argv[1]) as f:
    config = json.load(f)

# add treshold for number of example per concept
# and values to ignore
concepts = {}
for c_att in config["con_files"]:
    cdata = torch.load(c_att)
    for c in cdata:
        for att in config["concept_att"]:
            if att not in concepts: concepts[att] = []
            if type(c[att]) != list: concepts[att].append([c[att]])
            else: concepts[att].append(c[att])
            
targets=[]
tvalues=[]
for t_att in config["tar_files"]:
    tdata = torch.load(t_att)
    for t in tdata:
        targets.append(t)
        if t not in tvalues: tvalues.append(t)

concepts=[]
cvalues={}
for c_att in config["con_files"]:
    cdata = torch.load(c_att)
    for c in cdata:
        concepts.append(c)
        for ca in config["concept_att"]:
            if ca not in cvalues: cvalues[ca] = {}
            l=c[ca]
            if type(c[ca])!=list: l=[l]
            for v in l:
                if type(v) == torch.Tensor: v=float(v)
                if v not in cvalues[ca]: cvalues[ca][v] = 0
                cvalues[ca][v] += 1
                                
for li,layer in enumerate(config["layers"]):
    print(f"Layer {layer}:")
    acts = []
    lsom = ActSom(config["bsom_files"][li])
    # lsom.display()
    for actfile in config["act_files"]:
        pass
        act = torch.load(actfile)
        acts.extend(act[layer])
        break
    tacts = [[] for i in range(len(tvalues))]
    for ai,act in enumerate(acts):
        for vi,v in enumerate(tvalues):
            if targets[ai]==tvalues[vi]: tacts[vi].append(acts[ai])
    for vi,acts in enumerate(tacts):
        print(f"   target {tvalues[vi]} for layer {layer}")
        tsom = lsom.populate(tacts[vi])
        # tsom.display()
        tsom.save(f'{config["outdir"]}target_{tvalues[vi]}_{layer}.pickle')
    tacts=[]
    cacts={}
    for cai,ca in enumerate(config["concept_att"]):
        print(ca)
        cacts[ca] = {}
        for ai,act in enumerate(acts):        
            lcv = concepts[ai][ca]
            if type(lcv) != list: lcv=[lcv]
            for cv in lcv:
                if type(cv) == torch.Tensor: cv = float(cv)
                if cvalues[ca][cv] > config["concept_tresholds"][cai] and cv not in config["concept_ignore"][cai]:
                    if cv not in cacts[ca]: cacts[ca][cv] = []
                    cacts[ca][cv].append(act)
        for cv in cacts[ca]:
            print(f"   concept {ca}:{cv} for layer {layer}")
            tsom = lsom.populate(cacts[ca][cv])
            fcv = cv[cv.rindex("/")+1:] if type(cv) == str and "/" in cv else cv
            tsom.save(f'{config["outdir"]}concept_{ca}__{fcv}__{layer}.pickle')
