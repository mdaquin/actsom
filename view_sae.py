from argparse import ArgumentParser
import json, sys
import numpy as np
import utils as u
import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser(prog="view sae", description="visualiser for the SAE results of a layer")
parser.add_argument('configfile')
parser.add_argument('layer')
parser.add_argument('-o', '--output') # output image file
parser.add_argument('-hl', '--headless', action='store_true', default=False)
parser.add_argument('-c', '--concept')
parser.add_argument('-f', '--filter') # percentage of difference between concept and non-concept average activations to show

args = parser.parse_args()

config = json.load(open(args.configfile))

print("*** loading SOM results file ***")
df = pd.read_csv(config["som_results_file"])

print("*** loading SAE activations for layer ", args.layer, "***")
acts = json.load(open(config["sae_results_dir"]+"/"+args.layer+".json"))

# visualise average min, max activation for each unit
if "concept" in args and args.concept is not None: 
    column = args.concept.split(":")[0]
    value = args.concept.split(":")[1]
    print(f"looking at {column} with value {value}")
    ind = 0
    cacts = None
    ncacts = None
    values=[]
    for ab in acts: # kept the batches in run_dataset... should not have
        for a in ab:
            if value in str(df.iloc[ind][column]).lower(): 
                if df.iloc[ind][column] not in values: values.append(df.iloc[ind][column])
                if cacts is None: cacts = np.array(a).reshape((1,-1))
                else: cacts = np.concatenate((cacts, np.array(a).reshape((1,-1))), axis=0)
            else:
                if ncacts is None: ncacts = np.array(a).reshape((1,-1))
                else: ncacts = np.concatenate((ncacts, np.array(a).reshape((1,-1))), axis=0)
            ind += 1
    print(values)
    print("cacts", (cacts.shape if cacts is not None else None), "ncacts", ncacts.shape)

    avc = cacts.T.mean(axis=1)
    mac = cacts.T.max(axis=1)
    mic = cacts.T.min(axis=1)
    avnc = ncacts.T.mean(axis=1)
    manc = ncacts.T.max(axis=1)
    minc = ncacts.T.min(axis=1)
    if "filter" in args and args.filter is not None:
        filter = int(args.filter)
        filter = filter/100
        navc = []
        nmac = []
        nmic = []
        navnc = []
        nmanc = []
        nminc = []
        for i, a in enumerate(avc):
            if abs(a-avnc[i]) > filter*min(avnc[i], avc[i]):
                navc.append(a)
                nmac.append(mac[i])
                nmic.append(mic[i])
                navnc.append(avnc[i])
                nmanc.append(manc[i])
                nminc.append(minc[i])
        avc = np.array(navc)
        mac = np.array(nmac)
        mic = np.array(nmic)
        avnc = np.array(navnc)
        manc = np.array(nmanc)
        minc = np.array(nminc)
    plt.plot(avc, "r-")
    plt.plot(mac, "r--")
    plt.plot(mic, "r--")
    plt.plot(avnc, "b-")
    plt.plot(manc, "b--")
    plt.plot(minc, "b--")
    plt.title(f"average, min and max activations for each unit for {column}={value} (red) and for others (blue)")
    plt.xlabel("unit")
    plt.ylabel("activation")
    plt.show()
else: 
    npacts = None
    for ab in acts: # kept the batches in run_dataset... should not have
        noab = np.array(ab)
        if npacts is None: npacts = noab
        else: npacts = np.concatenate((npacts, noab), axis=0)
    plt.plot(npacts.T.mean(axis=1), "g-")
    plt.plot(npacts.T.max(axis=1), "g--")
    plt.plot(npacts.T.min(axis=1), "g--")
    plt.title("average, min and max activation for each unit")
    plt.xlabel("unit")
    plt.ylabel("activation")
    plt.show()
