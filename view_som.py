import sys
import os
from ksom import SOM
import dataset
import torch
from sklearn.decomposition import PCA
import pygame
import time
from argparse import ArgumentParser, FileType

def load_model(fn, device="cpu"):
    return torch.load(fn, map_location=device, weights_only=False)

def getLayer(model, layer):
    if "." in layer:
        first = layer[:layer.index(".")]
        rest  = layer[layer.index(".")+1:]
        nmod = getattr(model, first)
        return getLayer(nmod, rest)
    return getattr(model, layer) # won't work with subsubmodule

parser = ArgumentParser(prog="view SOM", description="visualiser for activation maps created through ActSOM")
parser.add_argument('somfile')
parser.add_argument('-o', '--output') # output image file
parser.add_argument('-n', '--num', action='store_true')
parser.add_argument('-ss', '--screensize', type=int, default=500)
parser.add_argument('-hl', '--headless', action='store_true', default=False)
parser.add_argument('-d', '--dataset') # show freas if no sample or concept
parser.add_argument('-m', '--model') # show freas if no sample or concept
parser.add_argument('-mc', '--modelcode') # show freas if no sample or concept
# parser.add_argument('-s', '--sample')
# parser.add_argument('-c', '--concept')

args = parser.parse_args()

print("loading SOM file")
som = torch.load(args.somfile)
print("MAP of size:", som.somap.shape)

# creating the display map
if "dataset" in args and args.dataset is not None:
    if "model" not in args or args.model is None:
       print("Error: need model as well as dataset, use -m or --model option")
       sys.exit(-1)
    if "modelcode" not in args or args.modelcode is None:
        print("Error: need the source code of the model class, use -mc or --modelcode option")
        sys.exit(-1)
    print("*** loading model")
    exec(open(args.modelcode).read())
    model=load_model(args.model)
    layer = args.somfile
    if "/" in layer: layer = layer[layer.rindex("/")+1:]
    if ".pt" in layer: layer = layer[:layer.rindex(".")]
    print("*** setup hook for layer", layer)
    global activation
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            if type(output) != torch.Tensor: activation[name] = output
            else: activation[name] = output.cpu().detach()
        return hook
    smod = getLayer(model, layer)
    smod.register_forward_hook(get_activation(layer))
    print("*** applying model and getting frequencies")
    som.to("cpu")
    dataset = dataset.KSDataset(args.dataset)
     # if concept 
     #     create the freq map
     #     or create the diff map
     # if element 
     #     create the distance map
     # else 
    rsom =  torch.tensor([0 for i in range(som.somap.shape[0])])
    for i in range(len(dataset)):
        print("   *** file", i)
        IS,OS = dataset[i]
        if IS.to(int).equal(IS): IS = IS.to(int)
        print("      *** applying SOM")
        PS = model(IS)
        if type(activation[layer]) == tuple: activation[layer] = activation[layer][0]
        acts = torch.flatten(activation[layer], start_dim=1).cpu()
        acts = (acts-som.minval.cpu())/(som.maxval.cpu()-som.minval.cpu())
        res = som(acts)[0]
        for r in res: rsom[r[0]*som.xs+r[1]] +=1 
    rsom = (rsom-rsom.min())/(rsom.max()-rsom.min())
else: 
    pca = PCA(n_components=3, random_state=42)
    rsom = pca.fit_transform(som.somap.detach().cpu())
    rsom = (rsom-rsom.min())/(rsom.max()-rsom.min()) # normalisation

screen_size=args.screensize # size of screen 
hl = "headless" in args and args.headless
pygame.font.init()
font = pygame.font.SysFont('Courrier', int((screen_size/som.xs)/4))
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
            color = (max(min(255, int(cs*255)), 0),
                     max(min(255, int(cs*255)), 0),
                     max(min(255, int(cs*255)), 0))
        else:
            color = (max(min(255, int(cs[0]*255)), 0),
                     max(min(255, int(cs[1]*255)), 0),
                     max(min(255, int(cs[2]*255)), 0))
        pygame.draw.rect(surface,
                         color,
                         pygame.Rect(x, y, unit, unit))
        if num and len(cs.shape)==0:
            if cs>0.5: tc = (50, 50, 50)
            else: tc = (200, 200, 200)
            texts = font.render(f"{float(cs):.3f}", False, tc)
            surface.blit(texts, (x+font.size("0")[0]*2,y+font.size("0")[0]*4))

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
