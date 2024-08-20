import sys
import os
from ksom import SOM
import torch
from sklearn.decomposition import PCA
import pygame
import time

if len(sys.argv) < 2:
    print("provide a file where a SOM model was saved. Optionally, you can provide a directory where to save the image of the SOM, in which case it won't be displayed")
    sys.exit(-1)

print("loading", sys.argv[1])
som = torch.load(sys.argv[1])
print("MAP of size:", som.somap.shape)


pca = PCA(n_components=3)
rsom = pca.fit_transform(som.somap)
rsom = (rsom-rsom.min())/(rsom.max()-rsom.min()) # normalisation

screen_size=600 # size of screen 
pygame.init()
surface = pygame.display.set_mode((screen_size,screen_size))

def display(somap, som_size):
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


display(rsom, som.xs)

while True:
     for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit()    
     time.sleep(0.1)
     pygame.display.flip()    
     pygame.display.update()
