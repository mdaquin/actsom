
from argparse import ArgumentParser
import json, sys
import numpy as np
import utils as u
import pandas as pd

parser = ArgumentParser(prog="view sae", description="find the concepts most represented by encoding neurons")
parser.add_argument('configfile')
parser.add_argument('layer')

args = parser.parse_args()

config = json.load(open(args.configfile))

print("*** loading SOM results file ***")
df = pd.read_csv(config["som_results_file"])

print("*** loading SAE activations for layer ", args.layer, "***")
acts = json.load(open(config["sae_results_dir"]+"/"+args.layer+".json"))

# make the list of all concepts, considering that the columns which 
# values are lists correspond to concepts.
concepts = {}
for dp in df.iloc:
    for c in dp.index:
        v=dp[c]
        if "['" in str(v): 
            v = str(v).replace("['",'["').replace("']",'"]').replace("',", '",').replace(" '", ' "')
            vl = json.loads(v)
            for vi in vl:
                if c+"="+vi not in concepts: 
                    concepts[c+"="+vi] = {"count": 0, "mact": np.zeros((len(acts[0][0])))}     
                concepts[c+"="+vi]["count"] += 1           
print("concepts", len(concepts), "found")
print("reduce to concepts that appear at least 10 times")
nconcepts = {}
for c in concepts:
    if concepts[c]["count"] >= 10: 
        nconcepts[c] = concepts[c]
concepts=nconcepts
print("concepts", len(concepts), "found")

ind = 0
for ab in acts: # kept the batches in run_dataset... should not have
    for a in ab:
        p = df.iloc[ind]
        for c in p.index:
            v=p[c]
            if "['" in str(v): 
                v = str(v).replace("['",'["').replace("']",'"]').replace("',", '",').replace(" '", ' "')
                vl = json.loads(v)
                for vi in vl:
                    if c+"="+vi in concepts:
                        concepts[c+"="+vi]["mact"] = concepts[c+"="+vi]["mact"]+np.array(a)
        ind+=1

# build the matrix concept x neuron
concept_ind = []
matrix = np.array([])
for c in concepts:
    nc = c
    if "/" in c: # just because sometimes it is a uri and too long to read
        nc = c[c.rindex("/")+1:]
    concept_ind.append(nc)
    if matrix.size == 0: matrix = concepts[c]["mact"]/concepts[c]["count"]
    else: matrix = np.vstack((matrix, concepts[c]["mact"]/concepts[c]["count"]))

# get the top 10 concept per neuron
for i, neuron in enumerate(matrix.T):
    ind = np.argpartition(neuron, -5)[-5:]
    print(f"{i}::", end=" ")
    for j in ind[::-1]:
        zscore1 = (neuron[j]-neuron.mean()) / neuron.std() # neuron is specific to this concept
        zscore2 = (neuron[j]-matrix[j].mean()) / matrix[j].std() # concept is specific to this neuron
        print(f"{concept_ind[j]}({zscore1:.3f}/{zscore2:.3f})", end=" ")
    print("\n")

print("*"*36)
print("*"*36)

# find the best zscore2 for each neuron, i.e. concept for which it is 
# most specific
for i, neuron in enumerate(matrix.T):
    zscore2 = (neuron-matrix[j].mean()) / matrix[j].std() # concept is specific to this neuron
    ind = np.argpartition(zscore2, -5)[-5:]
    print(f"{i}::", end=" ")
    for j in ind[::-1]:
        zscore1 = (neuron[j]-neuron.mean()) / neuron.std() # neuron is specific to this concept
        zscore2 = (neuron[j]-matrix[j].mean()) / matrix[j].std() # concept is specific to this neuron
        print(f"{concept_ind[j]}({zscore1:.3f}/{zscore2:.3f})", end=" ")
    print("\n")

print("*"*36)
print("*"*36)

# find (neuron, concept) pairs with the best harmonic mean of zscore1 and zscore2
hms = []
neur_conc = []
for i, neuron in enumerate(matrix.T):
    zscore1 = (neuron-neuron.mean()) / neuron.std() # neuron is specific to this concept
    zscore2 = (neuron-matrix[j].mean()) / matrix[j].std() # concept is specific to this neuron
    # harmonic mean of zscore1 and zscore2
    # hm = 2/(1/zscore1 + 1/zscore2)
    # hm = (2*abs(zscore1)*abs(zscore2))/(abs(zscore1)+abs(zscore2)) # trying with the F1 score formula
    hm = (zscore1+zscore2)/2 # mean
    for j,hmv in enumerate(hm):
        hms.append(hmv)
        neur_conc.append((i, j))
# 10 most representative neurons of concepts
ind = np.argsort(hms)
neurons = {}
for i in ind[::-1]:
    n, c = neur_conc[i]
    if hms[i] <= 2.0: break
    if n not in neurons: neurons[n] = []
    neurons[n].append(f"{concept_ind[c]} ({hms[i]:.3f})")
for n in neurons:
    print(f"{n}::", end=" ")
    for c in neurons[n]: print(c, end=" ")
    print("\n")