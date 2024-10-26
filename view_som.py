import sys
import os
from ksom import SOM
import torch
from sklearn.decomposition import PCA
import pygame
import time
from argparse import ArgumentParser, FileType

parser = ArgumentParser(prog="view SOM", description="visualiser for activation maps created through ActSOM")
parser.add_argument('somfile', type=FileType('rb'))
parser.add_argument('-o', '--output') # output image file
parser.add_argument('-n', '--num', action='store_true')
parser.add_argument('-ss', '--screensize', type=int, default=500)
parser.add_argument('-hl', '--headless', action='store_true', default=False)
parser.add_argument('-d', '--dataset') # show freas if no sample or concept
# parser.add_argument('-s', '--sample')
# parser.add_argument('-c', '--concept')

args = parser.parse_args()

print("loading SOM file")
som = torch.load(args.somfile)
print("MAP of size:", som.somap.shape)

# creating the display map
if "dataset" in args:
     # load dataset (see process dataset)
     # if concept 
     #     create the freq map
     #     or create the diff map
     # if element 
     #     create the distance map
     # else 
     #     create rsom with only frequencies...
     pass 
# else: 
pca = PCA(n_components=3)
rsom = pca.fit_transform(som.somap)
rsom = (rsom-rsom.min())/(rsom.max()-rsom.min()) # normalisation

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
        color = (max(min(255, int(cs[0]*255)), 0),
                 max(min(255, int(cs[1]*255)), 0),
                 max(min(255, int(cs[2]*255)), 0))
        pygame.draw.rect(surface,
                         color,
                         pygame.Rect(x, y, unit, unit))
    pygame.display.flip()
    pygame.display.update()
    if output:
        pygame.image.save(surface, output)
        print("saved to", output)

output = None
if "output" in args : output = args.output
display(rsom, som.xs, num=args.num, hl=hl, output=output)

if not hl: 
  while True:
     for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit()    
     time.sleep(0.1)
     pygame.display.flip()    
     pygame.display.update()
