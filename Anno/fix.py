import pandas as pd

df =pd.read_csv("list_landmarks_standardized_celeba.txt",delim_whitespace=True)
print(df.columns)
names = df["image_id"]
for i in range(len(names)):
    names.iloc[i]=str(names.iloc[i])[:-4]
print(names)
df["image_id"] = names
df.to_csv("list_landmarks_standardized_celeba.txt",index=False)