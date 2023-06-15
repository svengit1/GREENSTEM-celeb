# koristiti će se img_bboxed i njihovi odgovarajući handmade features
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset



class CelebADataset(Dataset):
    def __init__(self, image_folder, bbox_loc, landmark_loc, size_resized, mode="bbox", transform=None):
        self.size_resized = size_resized
        self.image_folder = image_folder
        self.bbox_file = bbox_loc
        self.mode = mode
        self.transform = transform
        self.image_files = os.listdir(image_folder)
        self.bbox_df = pd.read_csv(bbox_loc, delim_whitespace=True) if mode != "landm" else None
        self.landmark_df = pd.read_csv(landmark_loc, delim_whitespace=True) if mode != "bbox" else None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(image_file).convert('RGB')
        bbox, feats = None, None

        # Get bounding box for the current image
        if self.mode != "landm":
            row_bbox = self.bbox_df.loc[self.bbox_df['image_id'] == self.image_files[index]]
            bbox = list(row_bbox.values[0])[1:]
        if self.mode != "bbox":
            row_landmark = self.landmark_df.loc[self.landmark_df['image_id'] == self.image_files[index]]
            feats = list(row_landmark.values[0])[1:]

        image_size = image.size
        if self.transform:
            image = self.transform(image)
            bbox = torch.tensor(
                [self.size_resized[i % 2] * bbox[i] / image_size[i % 2] for i in range(len(bbox))]
                , dtype=torch.float32) if bbox else None

            feats = torch.tensor(
                [self.size_resized[i % 2] * feats[i] / image_size[i % 2] for i in range(len(feats))],
                dtype=torch.float32) if feats else None

        return image, bbox, feats

