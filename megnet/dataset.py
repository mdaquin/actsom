from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph
from pymatgen.core import Structure
from tqdm import tqdm
import pandas as pd 

# define a function to load the dataset
def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    data = pd.read_json("megnet/data/mp.2018.6.1.json")
    structures = []
    mp_ids = []

    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"], strict=False)):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)
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