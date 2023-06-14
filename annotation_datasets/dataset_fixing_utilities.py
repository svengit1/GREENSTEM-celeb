import pandas as pd


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

def shuffle_dataset(path,file,index_by="ID",seed=1):
    df = pd.read_csv(path + file)
    del df[index_by]
    df = df.sample(frac=1,
                   random_state=seed,
                   ).reset_index()
    df.to_csv("processed_SkinLabels_data.csv", index=False)

