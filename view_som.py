import sys
from ksom import SOM
import torch
from sklearn.decomposition import PCA
import pygame
import time
from argparse import ArgumentParser, FileType

parser = ArgumentParser(prog="view SOM", description="visualiser for activation maps created through ActSOM")
parser.add_argument('somfile')
parser.add_argument('-o', '--output') # output image file
parser.add_argument('-n', '--num', action='store_true')
parser.add_argument('-ss', '--screensize', type=int, default=500)
parser.add_argument('-hl', '--headless', action='store_true', default=False)

args = parser.parse_args()

print("loading SOM file")
som = torch.load(args.somfile, map_location="cpu")
print("MAP of size:", som.somap.shape)

# creating the display map
somap = som.somap.detach().cpu()
# somap = torch.nan_to_num(somap, 0.0)
if somap.isnan().any(): 
    print("SOM has nans...")
    sys.exit(-1)
if somap.shape[1] == 1:
    somap = somap.repeat(3, 3)
pca = PCA(n_components=3, random_state=42)
rsom = pca.fit_transform(somap)
if torch.tensor(rsom).isnan().any(): 
    print("PCA SOM has nans...")
    sys.exit(-1)
if rsom.min() != rsom.max():
    rsom = (rsom-rsom.min())/(rsom.max()-rsom.min()) # normalisation
if torch.tensor(rsom).isnan().any(): 
    print("Normalised PCA SOM has nans...")
    sys.exit(-1)

screen_size=args.screensize # size of screen 
hl = "headless" in args and args.headless
surface = pygame.display.set_mode((screen_size,screen_size))
if not hl:
    pygame.init()
    lname = sys.argv[1][sys.argv[1].rindex("/")+1:] if "/" in sys.argv[1] else sys.argv[1]
    lname = lname[:lname.rindex(".")] if "." in lname else lname
    pygame.display.set_caption("centroids for "+lname)

def display(somap, som_size, num=False, hl=False, output=None):
    if not hl:
      for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    unit = int(screen_size/som_size)
    for i,cs in enumerate(somap):
        x = int(i/som_size)
        y = i%som_size
        x = x*unit
        y = y*unit
        if len(cs.shape)==0: # freqs
            ncs = (cs-somap.min())/(somap.max()-somap.min())
            color = (max(min(255, int(ncs*255)), 0),
                     max(min(255, int(ncs*255)), 0),
                     max(min(255, int(ncs*255)), 0))
        else:
            color = (max(min(255, int(cs[0]*255)), 0),
                     max(min(255, int(cs[1]*255)), 0),
                     max(min(255, int(cs[2]*255)), 0))
        pygame.draw.rect(surface,
                         color,
                         pygame.Rect(x, y, unit, unit))
    if not hl:
        pygame.display.flip()
        pygame.display.update()
    if output:
        pygame.image.save(surface, output)
        print("saved to", output)

output = None
if "output" in args : output = args.output
display(rsom, som.xs, num=args.num, hl=hl, output=output)
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

parser = ArgumentParser(prog="view metrics", description="show computed metrics for a concept")
parser.add_argument('csvfile')

args = parser.parse_args()

df = pd.read_csv(args.csvfile)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.set_index("Unnamed: 0")
df = df[(df.KS_p < 0.5) | (df.MW_p < 0.5)]
df["MW"] = (df.MW - df.MW.min()) / (df.MW.max() - df.MW.min())
ax = df[["KL", "KS", "MW"]].plot(figsize=(10, 6))
ax.set_xticks(range(len(df)))
plt.xticks(list(df.index), rotation=45)
plt.show()
ax = df[["MW_p", "KS_p"]].plot()
ax.set_xticks(range(len(df)))
plt.xticks(list(df.index), rotation=45)
plt.show()
if not hl: 
  while True:
     for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit()    
     time.sleep(0.1)
     pygame.display.flip()    
     pygame.display.update()
