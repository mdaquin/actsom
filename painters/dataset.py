import torch 
import pandas as pd

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from sklearn.model_selection import train_test_split

class PainterDataset:
    def __init__(self, df, tokenizer, test=False):
        self.test = test
        self.texts = list(df['desc'].values)
        if self.test is False: self.labels = torch.tensor(df.inmuseum.values)
        self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.tokenizer.encode(text).ids
        text = torch.tensor(text, dtype=torch.long)
        if self.test is False:
              label = self.labels[idx]
              return text, label
        return text

def load_dataset(VOCAB_SIZE, split=False, SEED=42):
    print("*"*6,"loading json","*"*6)
    df = pd.read_json("painters/allpainters.json").T
    df["nbmuseum"] = df.nbmuseum.apply(lambda x: int(x))
    df["inmuseum"] = (df.nbmuseum > 0).apply(lambda x: float(x))

    print("*"*6,"balancing","*"*6)
    dfp=df[df.inmuseum==1.0]
    dfn=df[df.inmuseum==0.0].sample(len(dfp), random_state=SEED)
    df=pd.concat([dfp,dfn]).sample(frac=1)
    df=df.drop("nbmuseum", axis=1)

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(direction='left',length=512)
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE,min_frequency=2,special_tokens=['[PAD]','[UNK]'])
    tokenizer.train_from_iterator(df['desc'], trainer)
    torch.save(tokenizer, "tokenizer.pkl")

    if split:
        train_df, val_df = train_test_split(df, test_size=0.2,  shuffle=True, random_state = SEED)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_ds = PainterDataset(train_df, tokenizer)
        val_ds = PainterDataset(val_df, tokenizer)
        return train_ds, val_ds
    else:
        ds = PainterDataset(df, tokenizer)
        return ds

