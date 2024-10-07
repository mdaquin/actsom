import requests
import urllib
import json

query = '''
select ?painter ?desc ?category ?bcountry ?dcountry ?nat ?movement (count(distinct ?museum) as ?nbmuseum) {
    ?painter a <http://dbpedia.org/class/yago/Painter110391653>; 
              dbo:abstract ?desc;
              <http://purl.org/dc/terms/subject> ?category .
    optional {
        ?painter <http://dbpedia.org/ontology/birthPlace> [<http://dbpedia.org/ontology/country> ?bcountry].
    }
    optional {
        ?painter <http://dbpedia.org/ontology/deathPlace> [<http://dbpedia.org/ontology/country> ?dcountry].
    }
    optional {
        [<http://dbpedia.org/ontology/museum> ?museum] ?p ?painter .
    }
    optional {
        ?painter <http://dbpedia.org/property/nationality> ?nat .
    }
    optional {
        ?painter <http://dbpedia.org/ontology/movement> ?movement .
    }
    filter (lang(?desc) = "en")
} group by ?painter ?desc ?category ?bcountry ?dcountry ?nat ?movement limit __LIMIT__ offset __OFFSET__ 
'''

limit = 1000
offset = 0
done = False

data={}

while not done:
  params = {"query": query.replace("__LIMIT__", str(limit)).replace("__OFFSET__", str(offset)),
            "output": "json"}
  try:
    r = requests.get('https://dbpedia.org/sparql', params=params)
    results=r.json()
  except:
    print("failed")
    break
  count = 0
  for binding in results["results"]["bindings"]:
    count += 1
    painter = binding["painter"]["value"]
    if painter not in data:
        data[painter] = {"desc": binding["desc"]["value"],
                         "category": [binding["category"]["value"]],
                         "bcountry": [binding["bcountry"]["value"]] if "bcountry" in binding else ["unknown"],
                         "dcountry": [binding["dcountry"]["value"]] if "dcountry" in binding else ["unknown"],                         
                         "nat": [binding["nat"]["value"]] if "nat" in binding else ["unknown"],                         
                         "movement": [binding["movement"]["value"]] if "movement" in binding else ["unknown"],                         
                         "nbmuseum": binding["nbmuseum"]["value"]}
    else:
        if binding["category"]["value"] not in data[painter]["category"]:
            data[painter]["category"].append(binding["category"]["value"])                        
        if "bcountry" in binding and binding["bcountry"]["value"] not in data[painter]["bcountry"]:
            data[painter]["bcountry"].append(binding["bcountry"]["value"])
        if "dcountry" in binding and binding["dcountry"]["value"] not in data[painter]["dcountry"]:
            data[painter]["dcountry"].append(binding["dcountry"]["value"])
        if "nat" in binding and binding["nat"]["value"] not in data[painter]["nat"]:
            data[painter]["nat"].append(binding["nat"]["value"])            
        if "movement" in binding and binding["movement"]["value"] not in data[painter]["movement"]:
            data[painter]["movement"].append(binding["movement"]["value"])
        
  if count < limit: done = True
  offset = offset + limit
  print(offset, len(data.keys()))

with open("allpainters.json", "w") as f:
    json.dump(data,f, indent=2)
