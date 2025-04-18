from argparse import ArgumentParser
import json
from scipy.stats import entropy, ks_2samp, mannwhitneyu
import utils as u
import pandas as pd
import numpy as np

parser = ArgumentParser(prog="test concepts", description="measure de appearance of a concept in each layer")
parser.add_argument('configfile')
parser.add_argument('concept')

args = parser.parse_args()

config = json.load(open(args.configfile))

column = args.concept.split(":")[0]
value = args.concept.split(":")[1]

print(f"looking at {column} with value {value} in {config["model"]}")

# loading model and list layers in order
exec(open(config["modelclass"]).read())
model=u.load_model(config["model"])
layers = u.list_layers(model)

df = pd.read_csv(config["results_file"])
if type(df[column].iloc[0]) == str: 
        rdf = df[df.apply(lambda x: value.lower() in str(x[column]).lower(), axis=1)]
else: rdf = df[df[column] == float(value)]
print(f"found {len(rdf)}/{len(df)} objects of concept {column}={value}")

measures = {}
for layer in layers:
    # create overall frequency map for each layer
    vc = df[layer].value_counts()
    fmap = np.zeros(tuple(config["som_size"]))
    for i in vc.index: 
        fmap[i//fmap.shape[0], i%fmap.shape[0]] = vc[i]
    fmap = fmap/fmap.sum()
    # create concept frequency map for each layer
    vc = rdf[layer].value_counts()
    cmap = np.zeros(tuple(config["som_size"]))
    for i in vc.index: 
        cmap[i//cmap.shape[0], i%cmap.shape[0]] = vc[i]
    cmap = cmap/cmap.sum()
    # TODO: #10 gives some nan???
    klc = entropy(fmap.flatten(), qk=cmap.flatten(), base=2)
    ks2samp = ks_2samp(fmap.flatten(), cmap.flatten())
    mw = mannwhitneyu(fmap.flatten(), cmap.flatten())
    print(f"KL/KS/MW for {layer}: {klc:.2f}/{ks2samp.statistic:.2f}({ks2samp.pvalue:.2f})/{mw.statistic:.2f}({mw.pvalue:.2f})")
    measures[layer] = {"KL": klc, "KS": ks2samp.statistic, "KS_p": ks2samp.pvalue, "MW": mw.statistic, "MW_p": mw.pvalue}

pd.DataFrame(measures).T.to_csv(f"{config["results_file"]}_{column}_{value}_metrics.csv")

# compute max F1 for each layer... 
