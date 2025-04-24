# check the results when using the centroid of the 
# SOM cells as activations for the model
# only work with 1D layers
from argparse import ArgumentParser
import copy
import numpy as np
import json, sys, pygame, time, torch
import importlib as imp
import utils as u

parser = ArgumentParser(prog="som res", description="visualiser for frequency maps created through ActSOM")
parser.add_argument('configfile')
parser.add_argument('layer')
parser.add_argument('-o', '--output') # output image file
parser.add_argument('-ss', '--screensize', type=int, default=500)
parser.add_argument('-hl', '--headless', action='store_true', default=False)

args = parser.parse_args()
config = json.load(open(args.configfile))

screen_size=args.screensize # size of screen 
hl = "headless" in args and args.headless
surface = pygame.display.set_mode((screen_size,screen_size))
if not hl:
    pygame.init()
    lname = sys.argv[1][sys.argv[1].rindex("/")+1:] if "/" in sys.argv[1] else sys.argv[1]
    lname = lname[:lname.rindex(".")] if "." in lname else lname
    pygame.display.set_caption("centroids for "+lname)

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

print("Loading dataset...")
spec = imp.util.spec_from_file_location(config["datasetmodulename"], config["datasetclass"])
module = imp.util.module_from_spec(spec)
sys.modules[config["datasetmodulename"]] = module
spec.loader.exec_module(module)
exec("import "+config["datasetmodulename"])
data = eval(config["datasetcode"])
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

# load the som
layer = args.layer
print("Loading SOM for layer", args.layer)
som = torch.load(config["somdir"]+"/"+layer+".pt", weights_only=False)

# create a fake layer for each of the som cells
class SOMLayer(torch.nn.Module):
    def __init__(self, cell, min, max):
        super(SOMLayer, self).__init__()
        self.cell = (cell*(max-min)) + min
    def forward(self, x): 
        return self.cell.repeat(x.shape[0], 1)  
    
models_c = [[None for _ in range(som_size[1])] for _ in range(som_size[0])]
for i in range(som_size[0]):
    for j in range(som_size[1]):
        model_c = copy.deepcopy(model)
        u.set_module_by_name(model_c, layer, SOMLayer(som.somap[i][j], som.minval, som.maxval))
        models_c[i][j] = model_c

# get the output for each of the cells
outs = [[None for _ in range(som_size[1])] for _ in range(som_size[0])]
with torch.no_grad():
  for i in range(som_size[0]):
    for j in range(som_size[1]):
        print("Computing output for cell", i, j)
        som_out = []
        for data, _ in data_loader:
            data = data.to(device)
            out = models_c[i][j](data)
            outs[i][j] = out.cpu().numpy()[0]
            break

def text_display(outputs):
    for r in outputs:
        for c in r:
            print(c, end="\t|\t")
        print("\n")

def display(outputs, screen_size=400, num=False, hl=False, output=None):
    # make ouputs into a 2D numpy array
    outputs = np.array(outputs)
    if not hl:
      for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    unit = screen_size/outputs.shape[0]
    for i,r in enumerate(outputs):
        for j,c in enumerate(r):
            c = copy.deepcopy(c)
            x = int(i*unit)
            y = int(j*unit)
            if len(c) == 1: cs = [c[0], c[0], c[0]]
            elif len(c) == 2: cs = [c[0], c[1], 0]
            else: cs=c[:3]
            for k,v in enumerate(cs): 
                nv = (v-outputs.min())/(outputs.max()-outputs.min())
                cs[k] = max(min(255, int(nv*255)), 0)
            pygame.draw.rect(surface, cs, pygame.Rect(x, y, unit, unit))
    if not hl:
        pygame.display.flip()
        pygame.display.update()
    if output:
        pygame.image.save(surface, output)
        print("saved to", output)

text_display(outs)
display(outs, screen_size=screen_size)

if not hl: 
  while True:
     for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 sys.exit()    
     time.sleep(0.1)
     pygame.display.flip()    
     pygame.display.update()