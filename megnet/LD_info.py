import hashlib
import pickle
import urllib
from painters.dataset import load_dataset
from rdflib import Graph, URIRef, Literal
import time, os, sys

def get_fragment(u):
    if isinstance(u, Literal): return u.value
    if "#" in u: return u.split("#")[-1]
    if "/" in u: return u.split("/")[-1]
    return u

print("loading structs and mpdis")
with open("megnet/data/mp.2018.6.1_structures.pkl", "rb") as f:
    structures = pickle.load(f)
with open("megnet/data/mp.2018.6.1_mp_ids.pkl", "rb") as f:
    mp_ids = pickle.load(f)

print("loading kg data")
endpoint = 'http://bob:7200/repositories/TCKG'
baseuri = "https://k.loria.fr/ontologies/tckg/data/"

toignore = ["sameAs", "isPrimaryTopicOf", "prefLabel", "wikiPageID", "wikiPageUsesTemplate", "depiction", "wikiPageWikiLink", "wikiPageLength", "wikiPageRevisionID", "wasDerivedFrom", "wikiPageExternalLink", "label","comment", "type", "image", "thumbnail", "description", "abstract", "name", "givenName", "familyName", "birthDate", "deathDate", "birthPlace", "deathPlace"]
def filter(p):
    return p in toignore

def getURI(uri, ret=0):
    print(uri)
    g = Graph()
    try:
        g.parse(endpoint+"?query=describe+<"+uri+">")
    except Exception as e:
        print("!!!!!!! failed on", uri)
        print(e)
        sys.exit(1)
    return g

def get_ld_info(idx, nb, isuri=False):
    if not isuri:
        uri = "https://k.loria.fr/ontologies/tckg/data/"+mp_ids[idx]
    else: uri = idx
    g = getURI(uri)
    ret = {}
    for _, p, o in g.triples((URIRef(uri), None, None)):
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