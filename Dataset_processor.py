import pandas as pd
import torch
from torch import argmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from general_loader import process_image, process_image_bbox, bbox_model
from tools.datasets import CelebADatasetParent

default_model_input_size = (218, 218)


class CelebADatasetID(CelebADatasetParent):
    def __getitem__(self, index):
        image_name = self.image_files[index]
        return image_name


labels_df = []

batch_size = 1
loader_raw = CelebADatasetID("img_celeba", "")
loader_raw = DataLoader(loader_raw, batch_size=batch_size, shuffle=False)

path_raw = "img_celeba/"
model_race = torch.load("RACE_MODELS/modelEp6.pt").eval()
model_approp = torch.load("INAPPROP_MODELS/modelEp3.pt").eval()


def process(file, id):
    file = file[0].replace(".jpg", "")
    _, face_img = process_image_bbox(path_raw, file, bbox_model)
    inappropriate_label, _, _ = process_image(path_raw, file, model_approp)
    race_label, _, _ = process_image(path_raw, "", model_race, image=face_img)
    inappropriate_label = argmax(inappropriate_label).item()
    race_label = argmax(race_label).item()
    dct = {"ID": id, "name": file + ".jpg", "inappropriate": inappropriate_label, "dark_skin": race_label}
    labels_df.append(dct)


id = 0

for file in tqdm(loader_raw):
    process(file, id)
    id += 1
pd.DataFrame(labels_df).to_csv("Anno/custom/list_attr_quality_control.txt", index=False)
