import pickle
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph
from pymatgen.core import Structure
from tqdm import tqdm
import pandas as pd 
import os

# define a function to load the dataset
def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
     # load the mp_ids, structures and eform if exists
    if os.path.exists("megnet/data/mp.2018.6.1_structures.pkl"):
        with open("megnet/data/mp.2018.6.1_structures.pkl", "rb") as f:
            structures = pickle.load(f)
        with open("megnet/data/mp.2018.6.1_mp_ids.pkl", "rb") as f:
            mp_ids = pickle.load(f)
        with open("megnet/data/mp.2018.6.1_eform_per_atom.pkl", "rb") as f:
            eform_per_atom = pickle.load(f)    
        return structures, mp_ids, eform_per_atom
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
    return structures, mp_ids, data["formation_energy_per_atom"].tolist()


print("*** Loading dataset")
structures, mp_ids, eform_per_atom = load_dataset()

print("*** getting element list")
elem_list = get_element_list(structures)

print("*** setting up graph converter")
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)

print("*** create the megnet dataset")
mp_dataset = MGLDataset(
    structures=structures,
    labels={"Eform": eform_per_atom},
    converter=converter,
)

for data in mp_dataset:
    print(data)
    break