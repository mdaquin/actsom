
from argparse import ArgumentParser
import json, sys
import utils as u


parser = ArgumentParser(prog="view sae", description="visualiser for the SAE results of a layer")
parser.add_argument('configfile')
parser.add_argument('layer')
parser.add_argument('-o', '--output') # output image file
parser.add_argument('-hl', '--headless', action='store_true', default=False)
parser.add_argument('-c', '--concept')

args = parser.parse_args()

config = json.load(open(args.configfile))


print("*** loading model")
exec(open(config["modelclass"]).read())
model=u.load_model(config["model"])

# setup activation hooks
global activation
activation = {}
def get_activation(name):
    def hook(model, input, output):
            if type(output) != torch.Tensor: activation[name] = output
            else: activation[name] = output.cpu().detach()
    return hook

smod = u.getLayer(model, args.layer)
smod.register_forward_hook(get_activation(args.layer))

