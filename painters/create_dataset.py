import pandas as pd
import json
import torch
from torchtext.data.utils import get_tokenizer # TODO: replacement for torchtext which is deprecated ?
from collections import Counter
from torchtext.vocab import vocab
from torch.nn.functional import pad

print("*"*6,"loading json","*"*6)
df = pd.read_json("painters/allpainters.json").T
# print(df.columns)
df["nbmuseum"] = df.nbmuseum.apply(lambda x: int(x))
df["inmuseum"] = (df.nbmuseum > 0).apply(lambda x: float(x))
print("*"*6,"balancing","*"*6)
dfp=df[df.inmuseum==1.0]
dfn=df[df.inmuseum==0.0].sample(len(dfp), random_state=42)
df=pd.concat([dfp,dfn])
df=df.drop("nbmuseum", axis=1)
print("*"*6,"tokenizing","*"*6)
tokenizer = get_tokenizer('basic_english')
counter = Counter()
for text in df["desc"]:
  counter.update(tokenizer(text))
voc = vocab(counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
voc.set_default_index(voc['<unk>'])
print("  The length of the new vocab is", len(voc))
torch.save(voc, f"painters/voc_{len(voc)}")

print("*"*6,"saving","*"*6)
tosave = []
for i,row in enumerate(df.iloc):
  t = torch.tensor(voc(tokenizer(row.desc)))
  if len(t) < 100: t=pad(t,(0,100), value=voc['<PAD>'])
  if len(t) > 100: t=t[:100]
  tosave.append({"I": t.tolist(), 
                 "O": row["inmuseum"], 
                 "C": {"category": row["category"], 
                       "bcountry": row["bcountry"], 
                       "nat": row["nat"],
                       "movement": row["movement"]}})
  if (i+1)%1000 == 0: 
    with open(f"painters/dataset/dataset_{int((i+1)/1000)}.json", "w") as f: json.dump(tosave,f)
    tosave=[]
with open(f"painters/dataset/dataset_{int((i+1)/1000)+1}.json", "w") as f: json.dump(tosave,f)
