import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


class CelebADatasetParent(Dataset):
    def __init__(self, image_folder, data_file, image_transform=None, label_transform=None, delim_whitespace=True,
                 df_index_key="image_id"):
        self.image_folder = image_folder
        self.data_file = data_file
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.image_files = os.listdir(image_folder)
        self.df_index_key = df_index_key
        self.data_df = pd.read_csv(data_file, delim_whitespace=delim_whitespace)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        image_file = os.path.join(self.image_folder, img_name)
        image = Image.open(image_file).convert('RGB')
        row = self.data_df.loc[self.data_df[self.df_index_key] == img_name]
        labels = list(row.values[0])
        # further on the child object processes the query
        if self.label_transform:
            labels = torch.tensor(self.label_transform(labels, image))
        if self.image_transform:
            image = self.image_transform(image)
        return image, labels


class SubsetFactory:
    def __init__(self, train_size, val_size, test_size):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def __call__(self, dataset, shuffle=False):
        indices = np.arange(self.train_size + self.test_size + self.val_size)

        if len(dataset) < len(indices):
            raise ValueError("Dataset is too small to be partitioned!")

        if shuffle:
            random = np.random.default_rng()
            random.shuffle(indices)

        train_indices = indices[:self.train_size]
        val_indices = indices[self.train_size:self.train_size + self.val_size]
        test_indices = indices[self.train_size + self.val_size:self.train_size + self.val_size + self.test_size]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        return train_dataset, val_dataset, test_dataset
