from painters.dataset import load_dataset
from rdflib import Graph, URIRef, Literal

def get_fragment(u):
    if isinstance(u, Literal): return u.value
    if "#" in u: return u.split("#")[-1]
    if "/" in u: return u.split("/")[-1]
    return u

toignore = ["sameAs", "isPrimaryTopicOf", "prefLabel", "wikiPageID", "wikiPageUsesTemplate", "depiction", "wikiPageWikiLink", "wikiPageLength", "wikiPageRevisionID", "wasDerivedFrom", "wikiPageExternalLink", "label","comment", "type", "image", "thumbnail", "description", "abstract", "name", "givenName", "familyName", "birthDate", "deathDate", "birthPlace", "deathPlace"]
def filter(p):
    return p in toignore

cache={}
def get_from_cache(uri):
    if uri in cache: return cache[uri]
    return None
def set_cache(uri, data):
    cache[uri] = data

def get_ld_info(idx, nb, isuri=False):
    if not isuri:
        ds = load_dataset(3000, split=False, SEED=42, incuri=True)
        item = ds[idx]
        uri = item[2]
    else: uri = idx
    print(uri)
    g = get_from_cache(uri)
    if g is None:
      g = Graph()
      try:
        g.parse(uri)
        set_cache(uri, g)
      except:
        print("Error parsing URI:", uri)
        return {}
    ret = {}
    for s, p, o in g.triples((URIRef(uri), None, None)):
        p = get_fragment(p)
        if not filter(p): 
            if p not in ret: ret[p] = []
            no = get_fragment(o)
            if hasattr(no, "year"): no = no.year
            if no not in ret[p]: ret[p].append(no)
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