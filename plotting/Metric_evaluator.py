import matplotlib as plt
import os
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
from testing.cnn_class import CNN

size_resized = (218,218)
device = torch.device("cuda")
## TRANSFROMS RESIZED IDE OBRNUTO ; TJ. Y,X!
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size_resized[::-1])])
from torch.utils.data import DataLoader

batch_size = 16


class CelebADataset(Dataset):
    def __init__(self, image_folder, bbox_file, transform=None):
        self.image_folder = image_folder
        self.bbox_file = bbox_file
        self.transform = transform
        self.image_files = os.listdir(image_folder)
        self.bbox_df = pd.read_csv(bbox_file, delim_whitespace=True,
                                   dtype={'x_1': int, 'y_1': int, 'width': int, 'height': int},
                                   usecols=['image_id', 'x_1', 'y_1', 'width', 'height'])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(image_file).convert('RGB')

        # Get bounding box for the current image
        row = self.bbox_df.loc[self.bbox_df['image_id'] == self.image_files[index]]
        bbox = [int(row['x_1'].values[0]), int(row['y_1'].values[0]), int(row['width'].values[0]),
                int(row['height'].values[0])]
        # print("bbox for this image:", self.image_files[index], bbox)
        image_size = image.size
        if self.transform:
            image = self.transform(image)
            bbox = [size_resized[0]*bbox[0]/image_size[0], size_resized[1]*bbox[1]/image_size[1],
                   size_resized[0]*bbox[2]/image_size[0], size_resized[1]*bbox[3]/image_size[1]]
            bbox = torch.tensor(bbox, dtype=torch.float32)

        return image, bbox

image_dir = "./img_celeba"
bbox_dir = "./Anno/list_bbox_celeba.txt"
dataset = CelebADataset(image_dir, bbox_dir, transform=transform)

indices = np.arange(len(dataset))

train_size = 50000
val_size = 3000
test_size = 3000
test_indices = indices[train_size+val_size:train_size+val_size+test_size]

test_dataset = Subset(dataset, test_indices)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

models = [
    torch.load("BBOX_MODELS/modelEp0.pt"),
    torch.load("BBOX_MODELS/modelEp3.pt"),
    torch.load("BBOX_MODELS/modelEp6.pt"),
    torch.load("BBOX_MODELS/modelEp9.pt"),
    torch.load("BBOX_MODELS/modelEp12.pt")
    ]

[models[i].eval() for i in range(len(models))]

loss_function = nn.MSELoss()



def test_for_batch(test_dl,models,loss_fn):
    mean_losses = {}
    abs_losses = {}
    for mID,model in enumerate(models):
        model.eval()
        mean_loss = 0
        abs_loss = 0
        for inputs,labels in tqdm(test_dl):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            y_pred = model(inputs)
            loss = loss_fn(y_pred,labels)
            mean_loss += loss.item() / batch_size
            abs_loss += loss.item()
        mean_loss /= len(test_dl)
        print(f"test accuracy for model {mID}: mean {round(mean_loss,3)} abs {round(abs_loss,3)}")
        mean_losses[f"iteration_{mID*3}"] = mean_loss
        abs_losses[f"iteration_{mID*3}"] = abs_loss
    return mean_losses,abs_losses



if __name__ == '__main__':
    mean_losses,abs_losses =test_for_batch(test_dataloader,models,loss_function)
    mean_loss_file = open("plotting/mean_losses.json", "w")
    abs_loss_file = open("plotting/abs_losses.json", "w")
    json.dump(mean_losses,fp=mean_loss_file)
    json.dump(abs_losses,fp=abs_loss_file)
