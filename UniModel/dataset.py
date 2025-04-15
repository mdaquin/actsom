import torch 
import pandas as pd

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from sklearn.model_selection import train_test_split

# class to represent the dataset
class UniDataset:
    def __init__(self, df, tokenizer, test=False):
        self.test = test
        self.texts = list(df['Description'].values)
        if self.test is False: self.labels = df.scores_overall.values
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.tokenizer.encode(text).ids
        text = torch.tensor(text, dtype=torch.long)
        if self.test is False:
            label = self.labels[idx] / 5.
            label = torch.tensor(label, dtype=torch.float32)
            return text, label
        return text

def load_dataset(VOCAB_SIZE, split=False, SEED=42, retmeanstd=True):
    # processing the dataset
    df = pd.read_csv("UniModel/merge2.csv").dropna()
    print("Number of samples:", len(df))
    ## tokenizer for the texts
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(direction='left',length=512)
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE,min_frequency=2,special_tokens=['[PAD]','[UNK]'])
    tokenizer.train_from_iterator(df['Description'], trainer)
    torch.save(tokenizer, "tokenizer.pkl")
    if split:
        train_df, val_df = train_test_split(df, test_size=0.2,  shuffle=True, random_state = SEED)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        # normalisation of score
        tdfmean = train_df.scores_overall.mean()
        tdfstd = train_df.scores_overall.std()

        train_df["scores_overall"] = (train_df.scores_overall - tdfmean) / tdfstd
        val_df["scores_overall"] = (val_df.scores_overall - tdfmean) / tdfstd

        train_ds = UniDataset(train_df, tokenizer)
        val_ds = UniDataset(val_df, tokenizer)
        if retmeanstd: return train_ds, val_ds, tdfmean, tdfstd
        else: return train_ds, val_ds
    else:
         # normalisation of score
        tdfmean = df.scores_overall.mean()
        tdfstd = df.scores_overall.std()
        df["scores_overall"] = (df.scores_overall - tdfmean) / tdfstd
        ds = UniDataset(df, tokenizer)
        if retmeanstd: return ds, tdfmean, tdfstd
        else: return ds
