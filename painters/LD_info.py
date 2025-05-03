import hashlib
import pickle
import urllib
from painters.dataset import load_dataset
from rdflib import Graph, URIRef, Literal
import time, os

def get_fragment(u):
    if isinstance(u, Literal): return u.value
    if "#" in u: return u.split("#")[-1]
    if "/" in u: return u.split("/")[-1]
    return u

toignore = ["sameAs", "isPrimaryTopicOf", "prefLabel", "wikiPageID", "wikiPageUsesTemplate", "depiction", "wikiPageWikiLink", "wikiPageLength", "wikiPageRevisionID", "wasDerivedFrom", "wikiPageExternalLink", "label","comment", "type", "image", "thumbnail", "description", "abstract", "name", "givenName", "familyName", "birthDate", "deathDate", "birthPlace", "deathPlace"]
def filter(p):
    return p in toignore

baduris = ['de.dbpedia.org']

class Cache:
    def __init__(self, path):
        self.path = path
    def get(self, k):
        key = hashlib.sha1(str(k).encode("utf-8")).hexdigest()
        if os.path.exists(self.path+"/"+str(key)+".pkl"):
            with open(self.path+"/"+str(key)+".pkl", "rb") as f:
                data = pickle.load(f)
                if len(data)==0: 
                    print(" 0")
                    return None
                print(" *")
                return data
        print(" -")
        return None
    def set(self, key, value):
        # print("** Caching", key)
        key = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
        with open(self.path+"/"+str(key)+".pkl", "wb") as f:
            pickle.dump(value, f)

cache=Cache("entcache")
# def get_from_cache(uri):
#     if uri in cache: return cache[uri]
#     return None
# def set_cache(uri, data):
#     cache[uri] = data

def getURI(uri, ret=0):
    g = Graph()
    for s in baduris: 
        if s in uri: return g
    try:
        g.parse(uri)
        cache.set(uri, g)
    except Exception as e:
        print("Error parsing URI:", uri)
        print(e)
        print("trying describe query")
        try:
            # urlencode the uri 
            euri = urllib.parse.quote(uri, safe='')
            print("https://dbpedia.org/sparql?query=DESCRIBE+<"+euri+">")
            g.parse("https://dbpedia.org/sparql?query=DESCRIBE+<"+euri+">", format="turtle")
            cache.set(uri, g)
        except Exception as e:
            print("Error parsing URI with DESCRIBE query:", uri)
            print(e)
            if ret >= 10: return g
            time.sleep(30)
            print("Retrying...")
            return getURI(uri, ret+1)
    return g

def get_ld_info(idx, nb, isuri=False):
    if not isuri:
        ds = load_dataset(3000, split=False, SEED=42, incuri=True)
        item = ds[idx]
        uri = item[2]
    else: uri = idx
    print(uri, end=" ")
    g = cache.get(uri)
    if g is None:
      g = getURI(uri)
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