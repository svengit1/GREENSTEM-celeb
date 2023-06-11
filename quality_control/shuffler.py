import pandas as pd

dset = pd.read_csv("processed_SkinLabels_data.csv")
del dset["ID"]
dset = dset.sample(frac=1,
                   random_state=21,
                   ).reset_index()
del dset["index"]
dset.to_csv("processed_SkinLabels_data.csv")
