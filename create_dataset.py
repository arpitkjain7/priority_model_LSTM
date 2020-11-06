import json

filepath = "/Users/arpitkjain/Desktop/Data/SmartBox/code_base/documents/vendor-invoice updated.json"
data = json.load(open(filepath,"r"))
syn_list = []
synonyms = set()
x = open("data/syns.csv", "w+")

for keys in data.get("keys"):
    syns = keys.get("synonyms")
    synonyms.add(syns)
    syn_list.append(syns)
    x.write(syns)
x.close()
