import dataset
import rdflib

def get_ld_info(idx, nb):
    ds = dataset.load_dataset(3000, split=False, SEED=42, incuri=True)
    item = ds[idx]
    uri = item[2]
    print(uri)
    g = rdflib.Graph()
    g.parse(uri)
    for s, p, o in g.triples((URIRef(uri), None, None)):
        print(f"{s} {p} {o}")

if __name__ == "__main__":
    # test
    res = get_ld_info(2572, 2)
    print(res)