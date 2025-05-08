from argparse import ArgumentParser
import sys
import pandas as pd
import json, os, pickle
from sklearn.tree import DecisionTreeClassifier, export_text
    
parser = ArgumentParser(prog="Train DTs on actsom datasets", description="Create decision trees for each cell of a SOM")
parser.add_argument('configfile')
parser.add_argument('maxdepth', type=int)
parser.add_argument('-o', '--output') # output json file
parser.add_argument('-s', '--sample', type=int) # output json file
parser.add_argument('-d', '--drop') # output json file

args = parser.parse_args()
config = json.load(open(args.configfile))
maxdepth = args.maxdepth
dtmoddir = config["DTmodelsdir"]
dsdirs = config["actsomDSDir"]
results = {}
for f in os.listdir(dsdirs):
    print(f)
    df = pd.read_csv(dsdirs+"/"+f)
    # remove all the lignes with only 0s in df
    df = df.loc[(df.drop(columns=["target"])!=0).any(axis=1)]

    if "drop" in args and args.drop is not None: 
        df =df.drop(args.drop, axis=1)
    if "sample" in args and args.sample is not None:
        print(args.sample)
        sys.exit(1)
        df = df.sample(args.sample)
    # train a decision tree on the data
    X = df.drop(columns=["target"])
    y = df["target"]
    # print(y)
    clf = DecisionTreeClassifier(max_depth=maxdepth)
    clf.fit(X, y)
    accuracy = clf.score(X, y)
    results[f] = accuracy
    # display the tree as text
    print("***", f, "::", accuracy)
    print(export_text(clf, feature_names=X.columns.tolist()))
    model_path = dtmoddir+"/"+f.replace(".csv", ".model")
    with open(model_path, "wb") as f: pickle.dump(clf, f)
    print("Saved model to", model_path)
# sort the dict results by its value
results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
for f in results: print(f,"::", results[f])