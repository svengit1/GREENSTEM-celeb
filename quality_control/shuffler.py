import pandas as pd


##OBSOLETE, DATALOADER ISTO SHUFLLEA
def shuffle_dataset(path,file,index_by="ID",seed=1):
    df = pd.read_csv(path + file)
    del df[index_by]
    df = df.sample(frac=1,
                   random_state=seed,
                   ).reset_index()
    df.to_csv("processed_SkinLabels_data.csv", index=False)
