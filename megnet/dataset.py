import pickle
import random
import matgl
from pymatgen.core import Structure
from tqdm import tqdm
import pandas as pd 
import os, torch
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from dgl.data.utils import split_dataset
from matgl.utils.training import ModelLightningModule
import lightning as pl


def load_dataset(ret_mpids=False) -> tuple[list[Structure], list[str], list[float]]:
    if os.path.exists("megnet/data/mp.2018.6.1_structures.pkl"):
        with open("megnet/data/mp.2018.6.1_structures.pkl", "rb") as f:
            structures = pickle.load(f)
        with open("megnet/data/mp.2018.6.1_mp_ids.pkl", "rb") as f:
            mp_ids = pickle.load(f)
        with open("megnet/data/mp.2018.6.1_eform_per_atom.pkl", "rb") as f:
            eform_per_atom = pickle.load(f)    
    else:
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
        eform_per_atom = data["formation_energy_per_atom"].tolist()
    elem_list = get_element_list(structures)
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
    mp_dataset = MGLDataset(
        structures=structures,
        labels={"Eform": eform_per_atom},
        converter=converter,
    )
    train_data, val_data, test_data = split_dataset(
        mp_dataset,
        frac_list=[0.1, 0.4, 0.5],
        shuffle=True,
        random_state=42,
    )
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn_graph,
        batch_size=256,
        num_workers=0,
        shuffle=False
    )
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("*** loading model")
    model = matgl.load_model("megnet/model")
    lit_module = ModelLightningModule(model=model)
    print("*** loading dataset")
    loader1, loader2, loader3 = load_dataset(ret_mpids=False)
    trainer = pl.Trainer(accelerator="cpu")
    trainer.test(lit_module, dataloaders=loader1)
