import os
import pandas as pd
import torch
from PIL import Image
from torch import argmax
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from loader_quality_control_NEWEST import process_image, process_image_bbox, model_one

size_resized = (218, 218)
transform = transforms.Compose([
    transforms.Resize(size_resized),
    transforms.ToTensor(),
])


class CelebADatasetID(Dataset):
    def __init__(self, image_folder, label_transform=None, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = os.listdir(image_folder)
        self.label_transform = label_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_folder + "/" + self.image_files[index]).convert('RGB')
        labels = self.image_files[index]
        # image.show()
        if self.transform:
            image = self.transform(image)
        return image, labels


labels_df = []
images_raw = os.listdir("img_celeba")
images_bboxed = os.listdir("img_celeba_bboxed")

batch_size = 1
loader_raw = CelebADatasetID("img_celeba", "", transform=transform)
loader_raw = DataLoader(loader_raw, batch_size=batch_size, shuffle=False)

path_raw = "img_celeba/"
path_bboxed = "img_celeba_bboxed/"
model_race = torch.load("RACE_MODELS/modelEp6.pt").eval()
model_approp = torch.load("INAPPROP_MODELS/modelEp3.pt").eval()
global id
id = 0


def process(image, file):
    global id
    global labels_df
    file = file[0].replace(".jpg", "")
    _, face_img = process_image_bbox(path_raw, file, model_one)
    inappropriate_label, _, _ = process_image(path_raw, file, model_approp)
    race_label, _, _ = process_image(path_raw, "", model_race, image=face_img)
    inappropriate_label = argmax(inappropriate_label).item()
    race_label = argmax(race_label).item()
    dct = {"ID": id, "name": file + ".jpg", "inappropriate": inappropriate_label, "dark_skin": race_label}
    labels_df.append(dct)
    id += 1


for image, file in tqdm(loader_raw):
    process(image, file)
pd.DataFrame(labels_df).to_csv("Anno/custom/list_attr_quality_control.txt", index=False)
