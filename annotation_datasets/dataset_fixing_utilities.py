import pandas as pd


def double_image_id_suffix_fix(path, file, suffix, delim_whitespace=True, index=False):
    df = pd.read_csv(f"{path}/{file}", delim_whitespace=delim_whitespace)
    names = df["image_id"].values
    for i in range(len(names)):
        names[i] = str(names[i]).rstrip(suffix)
    df["image_id"] = names
    df.to_csv(f"{path}/{file}", index=index)


def csv_separator_replacer(path, file, old_sep: str, new_sep: str):
    df = pd.read_csv(f"{path}/{file}", delimiter=old_sep)
    df.to_csv(f"{path}/{file}", sep=new_sep, index=False)


def shuffle_dataset_inplace(path, file, index_by="ID", seed=1):
    df = pd.read_csv(path + file)
    del df[index_by]
    df = df.sample(frac=1,
                   random_state=seed,
                   ).reset_index()
    df.to_csv(path+file, index=False)
