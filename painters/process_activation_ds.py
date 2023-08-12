import pandas as pd
import json
import torch
import numpy as np

RS = 51

df = pd.read_json("allpainters.json").T
df = df[["desc", "nat", "movement", "category", "nbmuseum"]]
df["nbmuseum"] = df.nbmuseum.apply(lambda x: int(x))
df["inmuseum"] = df.nbmuseum > 0
df["inmuseum"] = df.inmuseum.apply(lambda x: float(x))
print(df.inmuseum.value_counts())
dfp=df[df.inmuseum==1.0]
dfn=df[df.inmuseum==0.0].sample(len(dfp), random_state=RS)
df=pd.concat([dfp,dfn])
print(df.inmuseum.value_counts())

import re
def preprocess(n):
  if re.match("http://dbpedia.org/resource/.*_people", n):
    return n[len("http://dbpedia.org/resource/"):-len("_people")]
  return n
df["nationality"] = df.nat.apply(lambda x: x[0] if len(x) == 1 else "other/multiple")
df["nationality"] = df.nationality.apply(preprocess)
replacements={
    "http://dbpedia.org/resource/United_States": "American",
    "http://dbpedia.org/resource/Germany" : "German",
    "German; naturalized American": "German-American",
    "http://dbpedia.org/resource/German-American": "German-American",
    "http://dbpedia.org/resource/Americans": "American",
    "http://dbpedia.org/resource/Italians": "Italian",
    "http://dbpedia.org/resource/Spain": "Spanish",
    "http://dbpedia.org/resource/United_Kingdom": "British",
    "http://dbpedia.org/resource/German_American": "German-American",
    "http://dbpedia.org/resource/Italy": "Italian",
    "English": "British",
    "http://dbpedia.org/resource/Netherlands": "Dutch",
    "http://dbpedia.org/resource/Russians": "Russian",
    "http://dbpedia.org/resource/France": "French",
    "http://dbpedia.org/resource/Mexico": "Mexican",
    "http://dbpedia.org/resource/Poland": "Polish",
    "http://dbpedia.org/resource/England": "British",
    "http://dbpedia.org/resource/Poles": "Polish",
    "http://dbpedia.org/resource/Croats": "Croatian",
    "http://dbpedia.org/resource/Dutch_Republic": "Dutch",
    "http://dbpedia.org/resource/Canadians": "Canadian",
    "http://dbpedia.org/resource/Belgium": "Belgian",
    "http://dbpedia.org/resource/Switzerland": "Swiss",
    "http://dbpedia.org/resource/Denmark": "Danish",
    "http://dbpedia.org/resource/China": "Chinese",
    "http://dbpedia.org/resource/Sweden": "Swedish",
    "http://dbpedia.org/resource/Finns": "Finnish",
    "http://dbpedia.org/resource/Germans": "German",
    "http://dbpedia.org/resource/Armenians": "Armenian",
    "http://dbpedia.org/resource/Argentina": "Argentinian",
    "http://dbpedia.org/resource/Hungary": "Hungarian",
    "http://dbpedia.org/resource/Serbs": "Serbian",
    "http://dbpedia.org/resource/Czechs": "Czech",
    "http://dbpedia.org/resource/Ukraine": "Ukrainian",
    "http://dbpedia.org/resource/Puerto_Rico": "Puerto Rican",
    "http://dbpedia.org/resource/Republic_of_Venice": "Venitian",
    "http://dbpedia.org/resource/Austrians": "Austrian",
    "http://dbpedia.org/resource/Ukrainians": "Ukrainian",
    "http://dbpedia.org/resource/Singaporean": "Singaporean",
    "http://dbpedia.org/resource/People_of_the_United_States": "American",
    "http://dbpedia.org/resource/USSR": "Russian",
    "http://dbpedia.org/resource/United_Kingdom_of_Great_Britain_and_Ireland": "British",
    "http://dbpedia.org/resource/Belgians": "Belgian",
    "http://dbpedia.org/resource/Australians": "Australian",
    "http://dbpedia.org/resource/Baltic_German": "German",
    "http://dbpedia.org/resource/Scotland": "British",
    "http://dbpedia.org/resource/People's_Republic_of_China": "China",
    "http://dbpedia.org/resource/Kingdom_of_France": "French",
    "http://dbpedia.org/resource/French_(people)": "French",
    "http://dbpedia.org/resource/Puerto_Rican_citizenship": "Puerto Rican",
    "http://dbpedia.org/resource/British_person": "British",
    "http://dbpedia.org/resource/Norway": "Norwegian",
    "http://dbpedia.org/resource/Swedes": "Swedish",
    "http://dbpedia.org/resource/Great_Britain": "British",
    "http://dbpedia.org/resource/African_American": "American",
    "http://dbpedia.org/resource/Austria": "Austrian",
    "http://dbpedia.org/resource/Greeks": "Greek",
    "http://dbpedia.org/resource/Republic_of_Ireland": "Irish",
    "http://dbpedia.org/resource/Mexicans": "Mexican",
    "http://dbpedia.org/resource/Northern_Netherlands": "Dutch",
    "http://dbpedia.org/resource/Argentine": "Argentinian",
    "http://dbpedia.org/resource/Bulgarians": "Bulgarian",
    "http://dbpedia.org/resource/Czech_Republic": "Czech",
    "http://dbpedia.org/resource/Finnish_nationality_law": "Finnish",
    "http://dbpedia.org/resource/Serb": "Serbian",
    "http://dbpedia.org/resource/New_Zealander": "New Zealander",
    "http://dbpedia.org/resource/Iraqis": "Iraqis",
    "http://dbpedia.org/resource/German_Democratic_Republic": "German",
    "http://dbpedia.org/resource/India": "Indian",
    "http://dbpedia.org/resource/Danes": "Danish",
    "http://dbpedia.org/resource/Netherlands_(terminology)": "Dutch",
    "http://dbpedia.org/resource/Philippine": "Filipino",
    "http://dbpedia.org/resource/Korea": "Korean",
    "http://dbpedia.org/resource/Slovenes": "Slovene",
    "http://dbpedia.org/resource/Hungarians": "Hungarian",
    "http://dbpedia.org/resource/Flemish": "Flemish",
    "http://dbpedia.org/resource/Turkey": "Turkish",
    "http://dbpedia.org/resource/Albanians": "Albanian",
    "http://dbpedia.org/resource/Pakistani": "Pakistani",
    "http://dbpedia.org/resource/Venice": "Venitian",
    "http://dbpedia.org/resource/Ireland": "Irish",
    "http://dbpedia.org/resource/Finland": "Finnish",
    "http://dbpedia.org/resource/Georgians": "Georgian",
    "http://dbpedia.org/resource/Estonians": "Estonian",
    "http://dbpedia.org/resource/Greece": "Greek",
    "http://dbpedia.org/resource/Wales": "Wales",
    "http://dbpedia.org/resource/Peruvians": "Peruvian",
    "http://dbpedia.org/resource/Russia": "Russian",
    "http://dbpedia.org/resource/Moroccans": "Morroccan",
    "http://dbpedia.org/resource/Russian_Empire": "Russian",
    "http://dbpedia.org/resource/Portugal": "Portugese",
    "http://dbpedia.org/resource/Paraguayan": "Paraguayan",
    "http://dbpedia.org/resource/Singapore": "Singaporean",
    "http://dbpedia.org/resource/Paris": "French",
    "http://dbpedia.org/resource/Argentines": "Argentinian",
    "http://dbpedia.org/resource/Dutch_(ethnic_group)": "Dutch",
    "http://dbpedia.org/resource/Egyptians": "Egyptian",
    "http://dbpedia.org/resource/Bangladesh" : "Bengali",
    "http://dbpedia.org/resource/Norwegians" : "Norwegian",
    "http://dbpedia.org/resource/Georgia_(country)" : "Georgian",
    "http://dbpedia.org/resource/Republic_of_China" : "Chinese",
    "http://dbpedia.org/resource/Dutch_People" : "Dutch",
    "http://dbpedia.org/resource/Israelis" : "Israeli",
    "http://dbpedia.org/resource/Israel" : "Israeli",
    "http://dbpedia.org/resource/Yugoslavs" : "Yugoslav",
    "http://dbpedia.org/resource/Flanders" : "Flemish",
    "http://dbpedia.org/resource/Soviet_Union" : "Russian",
    "http://dbpedia.org/resource/Brazilians" : "Brazilian",
    "http://dbpedia.org/resource/American_nationality_law" : "American",
    "http://dbpedia.org/resource/United_States_of_America" : "American",
    "http://dbpedia.org/resource/Philippines" : "Filipino",
    "http://dbpedia.org/resource/United_States_nationality_law" : "American",
    "http://dbpedia.org/resource/Slovakia" : "Slovak",
    "http://dbpedia.org/resource/Puerto_Ricans" : "Puerto Rican",
    "http://dbpedia.org/resource/Demographics_of_France" : "French",
    "http://dbpedia.org/resource/Arab_citizens_of_Israel" : "Israeli",
    "http://dbpedia.org/resource/Kingdom_of_Great_Britain" : "British",
    "http://dbpedia.org/resource/Albania" : "Albanian",
    "http://dbpedia.org/resource/New_Zealand" : "New Zealander",
    "http://dbpedia.org/resource/South_Korea" : "Korean",
    "http://dbpedia.org/resource/Demographics_of_Belgium" : "Belgian",
    "http://dbpedia.org/resource/Kingdom_of_Hungary" : "Hungarian",
    "http://dbpedia.org/resource/Norwegian_nationality_law" : "Norwegian",
    "http://dbpedia.org/resource/People_of_England" : "British",
    "http://dbpedia.org/resource/Lithuanians" : "Lithuanian",
    "http://dbpedia.org/resource/Permanent_residency_in_Canada" : "Canadian",
    "http://dbpedia.org/resource/Bangladeshi" : "Bengali",
    "http://dbpedia.org/resource/Polish_nationality_law" : "Polish",
    "http://dbpedia.org/resource/Uruguayan" : "Uruguayan",
    "http://dbpedia.org/resource/Venezuela" : "Venezuelan",
    "http://dbpedia.org/resource/Belarusians" : "Belarusian",
    "http://dbpedia.org/resource/Estonia" : "Estonian",
    "http://dbpedia.org/resource/Montenegro" : "Montenegrin",
    "http://dbpedia.org/resource/Spaniards" : "Spanish",
    "http://dbpedia.org/resource/British_subject" : "British",
    "http://dbpedia.org/resource/Republic_of_Armenia" : "Armenian",
    "http://dbpedia.org/resource/Romanians" : "Romanian",
    "http://dbpedia.org/resource/American_citizen" : "American",
    "http://dbpedia.org/resource/Koreans" : "Korean",
    "http://dbpedia.org/resource/Iran" : "Iranian",
    "http://dbpedia.org/resource/Indian_nationality_law" : "Indian",
    "http://dbpedia.org/resource/Icelanders" : "Icelander",
    "http://dbpedia.org/resource/Colombians" : "Colombian",
    "http://dbpedia.org/resource/Italian_People" : "Italian",
    "http://dbpedia.org/resource/Lithuania" : "Lithuanian",
    "http://dbpedia.org/resource/Southern_Netherlands" : "Dutch",
    "http://dbpedia.org/resource/Syrian" : "Syrian",
    "http://dbpedia.org/resource/South_Korean_nationality_law" : "Korean",
    "http://dbpedia.org/resource/Austrian_Empire" : "Austrian",
    "http://dbpedia.org/resource/Brazil" : "Brazilian",
    "http://dbpedia.org/resource/The_Netherlands" : "Dutch",
    "http://dbpedia.org/resource/Japan" : "Japanese",
    "http://dbpedia.org/resource/Kingdom_of_Italy" : "Italian",
    "http://dbpedia.org/resource/Moldova" : "Moldavian",
    "http://dbpedia.org/resource/Serbians" : "Serbian",
    "http://dbpedia.org/resource/People_of_Bangladesh" : "Bengali",
    "http://dbpedia.org/resource/Australian_nationality_law" : "Australian",
    "http://dbpedia.org/resource/United_States_Citizen" : "American",
    "http://dbpedia.org/resource/German" : "German",
    "http://dbpedia.org/resource/Tunisia" : "Tunisian",
    "http://dbpedia.org/resource/Russian_federation" : "Russian",
    "http://dbpedia.org/resource/Malta" : "Maltese",
    "http://dbpedia.org/resource/People_of_France" : "French",
    "http://dbpedia.org/resource/Cyprus" : "Cypriot",
    "http://dbpedia.org/resource/Emiratis" : "Emirati"
}
df["nationality"] = df.nationality.apply(lambda x: replacements[x] if x in replacements else x)
df["nationality"] = df.nationality.apply(lambda x: "other/multiple" if "http://dbpedia.org/" in x else x)
for k in df.nationality.value_counts().index:
  if "http" in k: print(k)
print(df.nationality.value_counts().head(20))

actdata = torch.load("actdata")

act={}
y=[]
c=[]

for i,idx in enumerate(actdata["idx"]):
    if actdata["targets"][i][1]!=df.iloc[idx].inmuseum: print("**** Warning ****")
    for k in actdata["activations"]:
        if k not in act: act[k] = []
        act[k].append(actdata["activations"][k][i])
    y.append(df.iloc[idx].inmuseum)
    c.append({"movement": df.iloc[idx].movement, "category": df.iloc[idx].category, "nationality": df.iloc[idx].nationality})

torch.save(y,"painters_y")
torch.save(c,"painters_c")
torch.save(act,"painter_act")
