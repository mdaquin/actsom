from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

parser = ArgumentParser(prog="view metrics", description="show computed metrics for a concept")
parser.add_argument('csvfile')

args = parser.parse_args()

df = pd.read_csv(args.csvfile, index_col=0)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(df)
df = df[(df.KS_p < 0.5) | (df.MW_p < 0.5)]
df["MW"] = (df.MW - df.MW.min()) / (df.MW.max() - df.MW.min())
ax = df[["KL", "KS", "MW"]].plot(figsize=(10, 6))
ax.set_xticks(range(len(df)))
# plt.xticks(list(df.index), rotation=45)
plt.show()
ax = df[["MW_p", "KS_p"]].plot()
ax.set_xticks(range(len(df)))
plt.xticks(list(df.index), rotation=45)
plt.show()