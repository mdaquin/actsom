import torch, sys, json
sys.path.insert(0, "../KSOM/src/ksom")
from ksom import SOM, cosine_distance, euclidean_distance, nb_gaussian, nb_linear, nb_ricker
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import pygame
from sklearn.decomposition import PCA

def display(map, xoffset=0):
    if map.shape[1] > 3:
        pca = PCA(n_components=3)
        somap = pca.fit_transform(map.detach())
    else: somap = map
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    unit = int(screen_size/som_size[0])
    for i,cs in enumerate(somap):
        x = int(i/som_size[0])
        y = i%som_size[0]
        x = (x*unit)+xoffset
        y = y*unit
        try : 
            color = (max(min(255, int(cs[0]*255)), 0),
                     max(min(255, int(cs[1]*255)), 0),
                     max(min(255, int(cs[2]*255)), 0))
        except: 
            print(cs*255)
            sys.exit(-1)
        pygame.draw.rect(surface,
                         color,
                         pygame.Rect(x, y, unit, unit))
        # if labels is not None:
        #      lab = findLabel(i, map, labels)
        #      cp = surface.get_at((int(x+label_offset+unit/20),y+int(unit/5)))
        #      cl = (200, 200, 200)
        #      if cp[0] > 100 : cl = (0, 0, 0)
        #      texts = font.render(lab, False, cl)
        #      surface.blit(texts, (x+label_offset+unit/20,y+int(unit/5)))

    pygame.display.flip()
    pygame.display.update()


def train_som(dataloader, training_log):
    with torch.no_grad():
            som = None
            for ep in range(1, nepochs+1):
                print("*** Epoch",ep,"***")
                freqmap = torch.zeros(som_size[0]*som_size[1])
                pbar = tqdm(dataloader, f"Training SOM for {layer}")
                for acts in pbar:
                    acts=acts.to(device)
                    if som is None:
                        sample = acts[:som_size[0]*som_size[1]]
                        mins = acts.min(dim=0).values.to(device)
                        maxs = acts.max(dim=0).values.to(device)
                        som = SOM(som_size[0], som_size[1], len(activations[layer][0]), 
                                  dist=distance,
                                  neighborhood_init=som_size[0]*1.0, 
                                  neighborhood_drate=0.00001*som_size[0], 
                                  zero_init=True, sample_init=sample,
                                  minval=mins, maxval=maxs, 
                                  alpha_init=alpha, alpha_drate=alpha_drate, 
                                  neighborhood_fct=neighb_func, 
                                  device=device, return_dist=True)
                    # normalisation
                    acts = ((acts-mins)/(maxs-mins))
                    change,count, dist = som.add(acts)
                    pbar.set_postfix({'ch': f'{change:.8f}', 'd': f'{dist:.8f}'})
                    training_log.append({"layer": layer, "epoch": ep, "count": count, "change": change, "distance": dist})
                    bmu, dists = som(acts)
                    for i in bmu: freqmap[i[0]*som_size[0]+i[1]] += 1
                    ffreqmap = (freqmap - freqmap.min())/(freqmap.max()-freqmap.min())
                    # freqmap = freqmap.view(len(freqmap), 1).repeat((1,3))
                    display(ffreqmap.view(len(freqmap), 1).repeat((1,3)), xoffset=screen_size)
                    display(som.somap.cpu())
                    # saving som
                    torch.save(som, f"{somdir}/{layer}_{som_size[0]}x{som_size[1]}.pth")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("provide configuration file (JSON)")
        sys.exit(1)
    conf = sys.argv[1]

    # base config
    screen_size = 600
    pygame.init()
    surface = pygame.display.set_mode((screen_size*2,screen_size))
    config = json.load(open(conf))
    if "seed" in config: torch.manual_seed(config["seed"])
    if "som_size" in config: som_size = config["som_size"]
    else:
        print("SOM size required in config")
        sys.exit(1)
    if "nepochs" in config: nepochs = config["nepochs"]
    else:
        print("nepochs required in config")
        sys.exit(1)
    if "batch_size" in config: batch_size = config["batch_size"]
    else: batch_size = 128
    if "distance" in config: distance = config["distance"]
    else: distance = "euclidean"
    if distance == "euclidean": distance = euclidean_distance
    elif distance == "cosine": distance = cosine_distance
    else: 
        print("unknwon distance", distance)
        sys.exit(1)
    if "alpha" in config: alpha=config["alpha"]
    else: alpha = 5e-3
    if "alpha_drate" in config: alpha_drate=config["alpha_drate"]
    else: alpha_drate = 1e-6
    if "neighb_func" in config: neighb_func = config["neighb_func"]
    else: neighb_func = "linear"
    if neighb_func == "linear": neighb_func=nb_linear
    elif neighb_func == "gaussian": neighb_func=nb_gaussian
    elif neighb_func == "ricker": neighb_func=nb_ricker
    else: 
        print("Unknown neighb_func", neighb_func)
        sys.exit(1)
    if "activationfile" in config: 
        data = torch.load(config["activationfile"], weights_only=False)
        if "activation_field" in config: 
            if config["activation_field"] in data: activations = data[config["activation_field"]]
            else:
                print("activation_field", config["activation_field"], "not found in activation file.")
                sys.exit(1)
        else: 
            print("activation_field required in config")
    else: 
        print("activationfile or activationcode required in config (only activationfile implemented)")
        sys.exit(1)
    if "somdir" in config: somdir = config["somdir"]
    else: 
        print("somdir required in configuration")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runcpu = "runcpu" in config and config["runcpu"]
    if runcpu: device = torch.device("cpu")
    if device == torch.device("cuda"): print("USING GPU")
    training_log = []
    for layer in activations: # make a train_som function and a train 1 epoch function
         dataloader = DataLoader(activations[layer], batch_size=batch_size, shuffle=True)
         train_som(dataloader, training_log)   
       
    # saving logs
    df = pd.DataFrame(training_log)
    df.to_csv("training_log.csv")

    df[["change", "distance"]].plot()
    plt.show()