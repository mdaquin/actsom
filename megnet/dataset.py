import pickle
import random
import matgl
from pymatgen.core import Structure
from tqdm import tqdm
import pandas as pd 
import os
import torch

class MGNetDataset:
    def __init__(self, structures, mp_ids, labels, return_mpids=False):
        self.structures = structures
        self.mp_ids = mp_ids
        self.labels = labels
        self.return_mpids = return_mpids

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = self.structures[idx]
        label = self.labels[idx]
        if self.return_mpids:
            mp_id = self.mp_ids[idx]
            return structure, label, mp_id
        return structure, label

def load_dataset(ret_mpids=False) -> tuple[list[Structure], list[str], list[float]]:
    if os.path.exists("megnet/data/mp.2018.6.1_structures.pkl"):
        with open("megnet/data/mp.2018.6.1_structures.pkl", "rb") as f:
            structures = pickle.load(f)
        with open("megnet/data/mp.2018.6.1_mp_ids.pkl", "rb") as f:
            mp_ids = pickle.load(f)
        with open("megnet/data/mp.2018.6.1_eform_per_atom.pkl", "rb") as f:
            eform_per_atom = pickle.load(f)    
        return MGNetDataset(structures, mp_ids, eform_per_atom, return_mpids=ret_mpids)
    data = pd.read_json("megnet/data/mp.2018.6.1.json")
    structures = []
    mp_ids = []
    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"], strict=False)):
        print("   *** getting structure", mid)
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)
    with open("megnet/data/mp.2018.6.1_structures.pkl", "wb") as f:
        pickle.dump(structures, f)
    with open("megnet/data/mp.2018.6.1_mp_ids.pkl", "wb") as f:
        pickle.dump(mp_ids, f)
    with open("megnet/data/mp.2018.6.1_eform_per_atom.pkl", "wb") as f:
        pickle.dump(data["formation_energy_per_atom"].tolist(), f)
    return MGNetDataset(structures, mp_ids, data["formation_energy_per_atom"].tolist(), return_mpids=ret_mpids)


if __name__ == "__main__":
    print("*** loading model")
    model = matgl.load_model("megnet/model")
    print("*** loading dataset")
    dataset = load_dataset(ret_mpids=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    idx = random.randint(0, len(dataset)-1)
    res=model.predict_structure(dataset[idx][0])
    print(float(res), dataset[idx][1])
