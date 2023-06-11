import re

import pandas as pd
from tqdm import tqdm


def doubleImageIdSuffixFix(path, file, suffix, delim_whitespace=True, index=False):
    df = pd.read_csv(f"{path}/{file}", delim_whitespace=delim_whitespace)
    names = df["image_id"].values
    for i in range(len(names)):
        names[i] = str(names[i]).rstrip(suffix)
    df["image_id"] = names
    df.to_csv(f"{path}/{file}", index=index)


def csvSeparatorReplacer(path, file, old_sep:str, new_sep:str):
    df = pd.read_csv(f"{path}/{file}", delimiter=old_sep)
    df.to_csv(f"{path}/{file}",sep=new_sep,index=False)

f = open("existing/list_attr_celeba.txt")
x = f.read()
f.close()
new = re.sub(" +"," ",x)
new = re.sub("-1","0",new)
print(new)
f = open("existing/list_attr_celeba.txt","w")
f.write(new)