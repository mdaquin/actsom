import pickle
import random
import matgl
from pymatgen.core import Structure
from tqdm import tqdm, trange
import pandas as pd 
import os, torch
from matgl.graph.data import  collate_fn_graph #, MGLDataLoader
from matgl.ext.pymatgen import get_element_list
from dgl.data.utils import split_dataset
from matgl.utils.training import ModelLightningModule
import lightning as pl

from matgl.graph.converters import GraphConverter
import dgl
import numpy as np 
from pymatgen.optimization.neighbors import find_points_in_spheres


from collections.abc import Callable

from dgl.dataloading import GraphDataLoader

from dgl.data import DGLDataset
import json
from dgl.data.utils import load_graphs, save_graphs
from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph
 

class MGLDataset(DGLDataset):
    """Create a dataset including dgl graphs."""

    def __init__(
        self,
        filename: str = "dgl_graph.bin",
        filename_lattice: str = "lattice.pt",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_state_attr: str = "state_attr.pt",
        filename_labels: str = "labels.json",
        include_line_graph: bool = False,
        converter: GraphConverter | None = None,
        threebody_cutoff: float | None = None,
        directed_line_graph: bool = False,
        structures: list | None = None,
        labels: dict[str, list] | None = None,
        directory_name: str = "MGLDataset",
        graph_labels: list[int | float] | None = None,
        clear_processed: bool = False,
        save_cache: bool = True,
        raw_dir: str = "./",
        save_dir: str = "./",
    ):
        """
        Args:
            filename: file name for storing dgl graphs.
            filename_lattice: file name for storing lattice matrixs.
            filename_line_graph: file name for storing dgl line graphs.
            filename_state_attr: file name for storing state attributes.
            filename_labels: file name for storing labels.
            include_line_graph: whether to include line graphs.
            converter: dgl graph converter.
            threebody_cutoff: cutoff for three body.
            directed_line_graph (bool): Whether to create a directed line graph (CHGNet), or an
                undirected 3body line graph (M3GNet)
                Default: False (for M3GNet)
            structures: Pymatgen structure.
            labels: targets, as a dict of {name: list of values}.
            directory_name: name of the generated directory that stores the dataset.
            graph_labels: state attributes.
            clear_processed: Whether to clear the stored structures after processing into graphs. Structures
                are not really needed after the conversion to DGL graphs and can take a significant amount of memory.
                Setting this to True will delete the structures from memory.
            save_cache: whether to save the processed dataset. The dataset can be reloaded from save_dir
                Default: True
            raw_dir : str specifying the directory that will store the downloaded data or the directory that already
                stores the input data.
                Default: current working directory
            save_dir : directory to save the processed dataset. Default: same as raw_dir.
        """
        self.filename = filename
        self.filename_lattice = filename_lattice
        self.filename_line_graph = filename_line_graph
        self.filename_state_attr = filename_state_attr
        self.filename_labels = filename_labels
        self.include_line_graph = include_line_graph
        self.converter = converter
        self.structures = structures or []
        self.labels = labels or {}
        for k, v in self.labels.items():
            self.labels[k] = v.tolist() if isinstance(v, np.ndarray) else v
        self.threebody_cutoff = threebody_cutoff
        self.directed_line_graph = directed_line_graph
        self.graph_labels = graph_labels
        self.clear_processed = clear_processed
        self.save_cache = save_cache
        super().__init__(name=directory_name, raw_dir=raw_dir, save_dir=save_dir)

    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not."""
        files_to_check = [
            self.filename,
            self.filename_lattice,
            self.filename_state_attr,
            self.filename_labels,
        ]
        if self.include_line_graph:
            files_to_check.append(self.filename_line_graph)
        return all(os.path.exists(os.path.join(self.save_path, f)) for f in files_to_check)

    def process(self):
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = len(self.structures)  # type: ignore
        graphs, lattices, line_graphs, state_attrs = [], [], [], []

        for idx in trange(num_graphs):
            structure = self.structures[idx]  # type: ignore
            graph, lattice, state_attr = self.converter.get_graph(structure)  # type: ignore
            graph.mpid = structure.mpid
            graphs.append(graph)
            lattices.append(lattice)
            state_attrs.append(state_attr)
            graph.ndata["pos"] = torch.tensor(structure.cart_coords)
            graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["bond_vec"] = bond_vec
            graph.edata["bond_dist"] = bond_dist
            if self.include_line_graph:
                line_graph = create_line_graph(graph, self.threebody_cutoff, directed=self.directed_line_graph)  # type: ignore
                for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                    line_graph.ndata.pop(name)
                line_graphs.append(line_graph)
            graph.ndata.pop("pos")
            graph.edata.pop("pbc_offshift")
        if self.graph_labels is not None:
            state_attrs = torch.tensor(self.graph_labels).long()
        else:
            state_attrs = torch.tensor(np.array(state_attrs), dtype=matgl.float_th)

        if self.clear_processed:
            del self.structures
            self.structures = []

        self.graphs = graphs
        self.lattices = lattices
        self.state_attr = state_attrs
        if self.include_line_graph:
            self.line_graphs = line_graphs
            return self.graphs, self.lattices, self.line_graphs, self.state_attr
        return self.graphs, self.lattices, self.state_attr

    def save(self):
        """Save dgl graphs and labels to self.save_path."""
        if self.save_cache is False:
            return

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.labels:
            with open(os.path.join(self.save_path, self.filename_labels), "w") as file:
                json.dump(self.labels, file)
        save_graphs(os.path.join(self.save_path, self.filename), self.graphs)
        torch.save(self.lattices, os.path.join(self.save_path, self.filename_lattice))
        torch.save(self.state_attr, os.path.join(self.save_path, self.filename_state_attr))
        if self.include_line_graph:
            save_graphs(os.path.join(self.save_path, self.filename_line_graph), self.line_graphs)

    def load(self):
        """Load dgl graphs from files."""
        self.graphs, _ = load_graphs(os.path.join(self.save_path, self.filename))
        self.lattices = torch.load(os.path.join(self.save_path, self.filename_lattice))
        if self.include_line_graph:
            self.line_graphs, _ = load_graphs(os.path.join(self.save_path, self.filename_line_graph))
        self.state_attr = torch.load(os.path.join(self.save_path, self.filename_state_attr))
        with open(os.path.join(self.save_path, self.filename_labels)) as f:
            self.labels = json.load(f)
        for i, g in self.graphs:
            g.mpid = self.labels["mpids"][i]

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        items = [
            self.graphs[idx],
            self.lattices[idx],
            self.state_attr[idx],
            {
                k: torch.tensor(v[idx], dtype=matgl.float_th)
                for k, v in self.labels.items()
                if not isinstance(v[idx], str)
            },
        ]
        # print(self.graphs[idx].mpid)
        if self.include_line_graph:
            items.insert(2, self.line_graphs[idx])
        return tuple(items)

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)

def MGLDataLoader(
    train_data: dgl.data.utils.Subset,
    val_data: dgl.data.utils.Subset,
    collate_fn: Callable | None = None,
    test_data: dgl.data.utils.Subset = None,
    **kwargs,
) -> tuple[GraphDataLoader, ...]:
    print(train_data[0][0].__dict__)
    print(train_data[0][0].mpid)
    train_loader = GraphDataLoader(train_data, shuffle=True, collate_fn=collate_fn, **kwargs)
    val_loader = GraphDataLoader(val_data, shuffle=False, collate_fn=collate_fn, **kwargs)
    if test_data is not None:
        test_loader = GraphDataLoader(test_data, shuffle=False, collate_fn=collate_fn, **kwargs)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader

class Structure2Graph(GraphConverter):
    """Construct a DGL graph from Pymatgen Structure."""

    def __init__(
        self,
        element_types: tuple[str, ...],
        cutoff: float = 5.0,
    ):
        """Parameters
        ----------
        element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
            constructed with the same dimensionality of features.
        cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, structure: Structure) -> tuple[dgl.DGLGraph, torch.Tensor, list | np.ndarray]:
        """Get a DGL graph from an input Structure.

        :param structure: pymatgen structure object
        :return:
            g: DGL graph
            lat: lattice for periodic systems
            state_attr: state features
        """
        numerical_tol = 1.0e-8
        pbc = np.array([1, 1, 1], dtype=np.int64)
        element_types = self.element_types
        lattice_matrix = structure.lattice.matrix
        cart_coords = structure.cart_coords
        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r=self.cutoff,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
        src_id, dst_id, images, bond_dist = (
            src_id[exclude_self],
            dst_id[exclude_self],
            images[exclude_self],
            bond_dist[exclude_self],
        )
        g, lat, state_attr = super().get_graph_from_processed_structure(
            structure,
            src_id,
            dst_id,
            images,
            [lattice_matrix],
            element_types,
            structure.frac_coords,
        )
        g.mpid = structure.mpid
        #print(g.__dict__)
        #print(type(g), type(lat), type(state_attr))
        return g, lat, state_attr

def load_dataset(ret_mpids=False, shuffle=True) -> tuple[list[Structure], list[str], list[float]]:
    if os.path.exists("megnet/data/mp.2018.6.1_structures.pkl"):
        with open("megnet/data/mp.2018.6.1_structures.pkl", "rb") as f:
            structures = pickle.load(f)
        with open("megnet/data/mp.2018.6.1_mp_ids.pkl", "rb") as f:
            mp_ids = pickle.load(f)
        with open("megnet/data/mp.2018.6.1_eform_per_atom.pkl", "rb") as f:
            eform_per_atom = pickle.load(f)
        for i, struct in enumerate(structures):
            struct.mpid = mp_ids[i]  
        for i, label in enumerate(eform_per_atom):
            eform_per_atom[i] = float(i)
        #print(len(eform_per_atom))
        #print(len(mp_ids))
        #fuck = you
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
        structures=structures[:100],
        labels={"Eform": eform_per_atom[:100], "mpids": mp_ids[:100]},
        converter=converter
    )
    print(mp_dataset[0])
    print(mp_dataset[0][0].mpid)
    train_data, val_data, test_data = split_dataset(
        mp_dataset,
        frac_list=[0.1, 0.4, 0.5],
        shuffle=shuffle,
        random_state=42,
    )
    print(train_data[0])
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn_graph,
        batch_size=1,
        num_workers=0
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
