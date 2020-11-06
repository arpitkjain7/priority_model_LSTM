import pandas as pd
import random

data = pd.read_csv("data/orig_key.csv")
f = open("data/orig_key_updated_7.csv", "w+")
for idx, item in enumerate(data.values):
    list_key = []
    list_label = []
    for i in range(3):
        if i+idx < len(data["DATA"]):
            list_key.append(data["DATA"][idx + i])
            list_label.append(data["LABEL"][idx + i])
    print(list_key)
    f.write(" ".join(list_key)+","+" ".join(list_label)+'\n')
