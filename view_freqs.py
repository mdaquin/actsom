from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json, sys, pygame, time

parser = ArgumentParser(prog="view freq", description="visualiser for frequency maps created through ActSOM")
parser.add_argument('configfile')
parser.add_argument('layer')
parser.add_argument('-o', '--output') # output image file
parser.add_argument('-n', '--num', action='store_true')
parser.add_argument('-ss', '--screensize', type=int, default=500)
parser.add_argument('-hl', '--headless', action='store_true', default=False)
parser.add_argument('-c', '--concept')

args = parser.parse_args()

config = json.load(open(args.configfile))

print("opening", config["results_file"])

df = pd.read_csv(config["results_file"])
vc = df[args.layer].value_counts()
fmap = np.zeros(tuple(config["som_size"]))
for i in vc.index: 
    fmap[i//fmap.shape[0], i%fmap.shape[0]] = vc[i]
fmap = fmap/fmap.sum()

screen_size=args.screensize # size of screen 
hl = "headless" in args and args.headless
pygame.font.init()
font = pygame.font.SysFont('Courrier', int((screen_size/fmap.shape[1])/4))
surface = pygame.display.set_mode((screen_size,screen_size))
if not hl:
    pygame.init()
    pygame.display.set_caption("frequencies for "+args.layer)

def display(fmap, som_size, hl=False, output=None):
    if not hl:
      for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    unit = int(screen_size/som_size)
    for i,l in enumerate(fmap):
        for y,cs in enumerate(l):
            print(i,y)
            x = i*unit
            y = y*unit
            ncs = (cs-fmap.min())/(fmap.max()-fmap.min())
            color = (max(min(255, int(ncs*255)), 0),
                     max(min(255, int(ncs*255)), 0),
                     max(min(255, int(ncs*255)), 0))
            pygame.draw.rect(surface,
                         color,
                         pygame.Rect(x, y, unit, unit))
            ncs = (cs-fmap.min())/(fmap.max()-fmap.min())
            if ncs>0.5: tc = (50, 50, 50)
            else: tc = (200, 200, 200)
            texts = font.render(f"{float(cs)*100:02.2f}%", False, tc)
            surface.blit(texts, (x+font.size("0")[0]*2,y+font.size("0")[0]*4))
    pygame.display.flip()
    pygame.display.update()
    if output:
        pygame.image.save(surface, output)
        print("saved to", output)

output = None
if "output" in args : output = args.output
display(fmap, fmap.shape[0], hl=hl, output=output)

if not hl: 
  while True:
     for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit()    
     time.sleep(0.1)
     pygame.display.flip()    
     pygame.display.update()