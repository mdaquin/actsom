import dataset
from rdflib import Graph, URIRef, Literal

def get_fragment(u):
    if isinstance(u, Literal): return u.value
    if "#" in u: return u.split("#")[-1]
    if "/" in u: return u.split("/")[-1]
    return u

toignore = ["sameAs", "isPrimaryTopicOf", "prefLabel", "wikiPageID", "wikiPageUsesTemplate", "depiction", "wikiPageWikiLink", "wikiPageLength", "wikiPageRevisionID", "wasDerivedFrom", "wikiPageExternalLink", "label","comment", "type", "image", "thumbnail", "description", "abstract", "name", "givenName", "familyName", "birthDate", "deathDate", "birthPlace", "deathPlace"]
def filter(p):
    return p in toignore

def get_ld_info(idx, nb, isuri=False):
    if not isuri:
        ds = dataset.load_dataset(3000, split=False, SEED=42, incuri=True)
        item = ds[idx]
        uri = item[2]
    else: uri = idx
    print(uri)
    g = Graph()
    g.parse(uri)
    ret = {}
    for s, p, o in g.triples((URIRef(uri), None, None)):
        p = get_fragment(p)
        if not filter(p): 
            if p not in ret: ret[p] = []
            if o not in ret[p]: ret[p].append(get_fragment(o))
            if isinstance(o, URIRef) and nb>1:
                ret2 = get_ld_info(o, nb-1, isuri=True)
                for k, v in ret2.items():
                    if p+"__"+k not in ret: ret[p+"__"+k] = []
                    for i in v:
                        if i not in ret[p+"__"+k]: ret[p+"__"+k].append(i)
    return ret

if __name__ == "__main__":
    # test
    res = get_ld_info(2572, 3)
    print(res)