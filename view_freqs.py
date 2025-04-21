from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json, sys, pygame, time, torch
import importlib as imp
import utils as u

parser = ArgumentParser(prog="view freq", description="visualiser for frequency maps created through ActSOM")
parser.add_argument('configfile')
parser.add_argument('layer')
parser.add_argument('-o', '--output') # output image file
parser.add_argument('-n', '--num', action='store_true')
parser.add_argument('-ss', '--screensize', type=int, default=500)
parser.add_argument('-hl', '--headless', action='store_true', default=False)

args = parser.parse_args()
config = json.load(open(args.configfile))

torch.manual_seed(config["seed"])
som_size = config["som_size"]
base_som_dir = config["somdir"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "runcpu" in config: device = torch.device("cpu")
if device == torch.device("cuda"): print("USING GPU")

with torch.no_grad():
    print("Loading model...")   
    # exec(open(config["modelclass"]).read())
    spec = imp.util.spec_from_file_location(config["modelmodulename"], config["modelclass"])
    module = imp.util.module_from_spec(spec)
    sys.modules[config["modelmodulename"]] = module
    spec.loader.exec_module(module)
    model = u.load_model(config["model"], device=device)
    model.eval()

print("Setting up activation hooks...")
u.activation = {}
list_layers = u.set_up_activations(model)

print("Loading dataset...")
spec = imp.util.spec_from_file_location(config["datasetmodulename"], config["datasetclass"])
module = imp.util.module_from_spec(spec)
sys.modules[config["datasetmodulename"]] = module
spec.loader.exec_module(module)
exec("import "+config["datasetmodulename"])
data = eval(config["datasetcode"])
data_loader = torch.utils.data.DataLoader(data, batch_size=config["batchsize"], shuffle=True)

print("Loading SOM for layer", args.layer)
som = torch.load(config["somdir"]+"/"+args.layer+".pt", weights_only=False)

print("Applying model and SOM")
fmap = np.zeros(tuple(config["som_size"]), dtype=np.float32)
fmap[:] = 0
with torch.no_grad():
    for input, target in data_loader:
        u.activation = {}
        input = input.to(device)
        p = model(input)
        acts = u.activation[args.layer]
        if type(acts) == tuple: acts = acts[0]
        if config["aggregation"] == "flatten": 
            acts = torch.flatten(acts, start_dim=1).to(device)
        elif config["aggregation"] == "mean":
                    if len(acts.shape) > 2:
                        acts = torch.mean(acts, dim=1).to(device)
                    else: acts = acts.to(device)
        else: 
            print("unknown aggregation, check config")
            sys.exit(-1)
        acts = (acts-som.minval)/(som.maxval-som.minval) # we don't do this for SAE combination...
        res = som(acts)
        for bmu in res[0]:
            fmap[bmu[0].item(), bmu[1].item()] += 1
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
        for j,cs in enumerate(l):
            x = i*unit
            y = j*unit
            ncs = (cs-fmap.min())/(fmap.max()-fmap.min())
            gb = max(min(255, int(ncs*255)), 0)
            r = max(min(255, int(ncs*255)), 0)
            color = (r,gb,gb)
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