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
        if self.test is False: self.labels = torch.tensor(df.scores_overall.values)
        self.labels = torch.nn.functional.one_hot(self.labels.to(torch.long), num_classes=3).float() # needs to go in the dataset
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.tokenizer.encode(text).ids
        text = torch.tensor(text, dtype=torch.long)
        if self.test is False:
              label = self.labels[idx]
              return text, label
        return text

def load_dataset(VOCAB_SIZE, split=False, SEED=42, retquantile=False):
    # processing the dataset
    df = pd.read_csv("UniModel_classif/merge2.csv").dropna()
    print("Number of samples:", len(df))
    ## tokenizer for the texts
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(direction='left',length=512)
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE,min_frequency=2,special_tokens=['[PAD]','[UNK]'])
    tokenizer.train_from_iterator(df['Description'], trainer)
    torch.save(tokenizer, "tokenizer.pkl")

    # obtain quantile for the scores
    q1 = df.scores_overall.quantile(0.3333)
    q2 = df.scores_overall.quantile(0.6667)

    if split:
        train_df, val_df = train_test_split(df, test_size=0.2,  shuffle=True, random_state = SEED)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_df2 = train_df.copy()
        train_df2.loc[train_df.scores_overall <= q1, 'scores_overall'] = 0
        train_df2.loc[(train_df.scores_overall > q1) & (train_df.scores_overall <= q2), 'scores_overall'] = 1
        train_df2.loc[train_df.scores_overall > q2, 'scores_overall'] = 2

        test_df2 = val_df.copy()
        test_df2.loc[val_df.scores_overall <= q1, 'scores_overall'] = 0
        test_df2.loc[(val_df.scores_overall > q1) & (val_df.scores_overall <= q2), 'scores_overall'] = 1
        test_df2.loc[val_df.scores_overall > q2, 'scores_overall'] = 2

        train_ds = UniDataset(train_df2, tokenizer)
        val_ds = UniDataset(test_df2, tokenizer)
        if retquantile: return train_ds, val_ds, q1, q2
        else: return train_ds, val_ds
    else:
        df2 = df.copy()
        df2.loc[df.scores_overall <= q1, 'scores_overall'] = 0
        df2.loc[(df.scores_overall > q1) & (df.scores_overall <= q2), 'scores_overall'] = 1
        df2.loc[df.scores_overall > q2, 'scores_overall'] = 2

        ds = UniDataset(df, tokenizer)
        if retquantile: return ds, q1, q2
        else: return ds
