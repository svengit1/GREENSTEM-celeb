import pandas as pd
import numpy as np
import os

import torch
from sympy.physics.control.control_plots import matplotlib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv(r'C:\Users\Korisnik\OneDrive\Documents\sven_img\x.csv',nrows=1505, usecols=['Name','lefteye_x','lefteye_y','righteye_x','righteye_y','nose_x','nose_y','leftmouth_x','leftmouth_y','rightmouth_x','rightmouth_y'])

df.head()
print(df)
data=tf.convert_to_tensor(df)
num_train_examples = int(df.shape[0]*0.8)
df_train = df.iloc[:num_train_examples, :]
df_test = df.iloc[num_train_examples:, :]

df_train.to_csv('celeba_gender_attr_train.txt', sep=" ")
df_test.to_csv('celeba_gender_attr_test.txt', sep=" ")
img = Image.open(r'C:\Users\Korisnik\OneDrive\Documents\sven_img\celebA_mini\000001.jpg')
print(df_train,df_test)
plt.imshow(img)
plt.show()
print(data)