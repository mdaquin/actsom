from argparse import ArgumentParser
import json, sys, os
import importlib.util as imp_util
import pandas as pd

# take a config and a dataset 
# create the dataset based on LD completion

class Cache:
    def __init__(self, path):
        self.path = path
    def get(self, key):
        if os.path.exists(self.path+"/"+str(key)+".json"):
            with open(self.path+"/"+str(key)+".json") as f:
                data = json.load(f)
                if len(data)==0: 
                    print("** empty cache for", key)
                    return None
                print("** got from cache", key)
                return data
        print("** no cache for", key)
        return None
    def set(self, key, value):
        print("** Caching", key)
        with open(self.path+"/"+str(key)+".json", "w") as f:
            json.dump(value, f)

def dealWithNumerical(df):
    for col in df.columns:
        for i,v in enumerate(df[col]):
            if type(v) == list and len(v) > 0:
                if type(v[0]) != str:
                    print("##", col, i, v)
                elif v[0].replace('.','',1).isdigit():
                    print("##.##", col, i, v)
    return df

parser = ArgumentParser(prog="view freq", description="visualiser for frequency maps created through ActSOM")
parser.add_argument('configfile')
parser.add_argument('actsom_dataset')

args = parser.parse_args()
config = json.load(open(args.configfile))
cache = Cache(config["ldcache"])

print("** Loading LDINfo module")
spec = imp_util.spec_from_file_location(config["ldinfomodulename"], config["ldinfoscript"])
module = imp_util.module_from_spec(spec)
sys.modules[config["ldinfomodulename"]] = module
spec.loader.exec_module(module)
exec("import "+config["ldinfomodulename"])

print("** Loading actsom dataset")
ds = json.load(open(args.actsom_dataset))
layer = args.actsom_dataset.split("/")[-1].split(".")[0]

for ci,cell in enumerate(ds):
    hcs = []
    lcs = []
    for hcidx in cell["hc"]:
        cached = cache.get(hcidx)
        if cached is None:
            info = eval(config["ldinfocode"].replace("[[IDX]]", str(hcidx)))
            cache.set(hcidx, info)
            info["target"] = 1
            hcs.append(info)
        else: 
            cached["target"] = 1
            hcs.append(cached)
    for hcidx in cell["lc"]:
        cached = cache.get(hcidx)
        if cached is None:
            info = eval(config["ldinfocode"].replace("[[IDX]]", str(hcidx)))
            cache.set(hcidx, info)
            info["target"] = 0
            lcs.append(info)
        else: 
            cached["target"] = 0
            lcs.append(cached)
    df = pd.DataFrame(hcs+lcs)
    # remove columns with more than 50% missing values
    df = df.dropna(axis=1, thresh=len(df)*0.5) # 0.5 should be a parameter in config
    # columns with numerical values should be averaged...
    df = dealWithNumerical(df) # forget it for now, there is none...
    # multi-value columns should be exploded
    for col in df.columns:
        nna = df[col].dropna()
        if len(nna) > 0  and type(nna.iloc[0]) == list:
            one_hot = pd.get_dummies(df[col].apply(pd.Series).stack(), prefix=col, prefix_sep="::").groupby(level=0).sum()
            df = df.drop(col, axis=1).join(one_hot, how='left')
        else:
            if col != "target": print("***", col, "not exploded")
    df = df.fillna(0) # not ideal if we had numerical stull, but...
    df.to_csv(config["actsomDSDir"]+"/"+layer+"_"+str(ci)+".csv", index=False)