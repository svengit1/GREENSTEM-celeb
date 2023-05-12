import os
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image
from CelebADrawer import BboxDrawer
from testing.cnn_class import CNN
import pandas as pd
import time



class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.act4 = nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc4 = nn.Linear(608256, 4)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)

        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.act3(self.conv3(x))
        x = self.pool3(x)

        x = self.act4(self.conv4(x))
        x = self.flatten(x)
        x = self.fc4(x)
        return x


model = torch.load("BBOX_MODELS/modelEp9.pt")
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([218, 218])])

path= "img_celeba/"
for i in range(40000,40010):
    image = Image.open(path+f"0{i}.jpg").convert('RGB')
    image_resized = transforms.Resize([218, 218])(image)
    img_transformed = torch.unsqueeze(transform(image),0)
    model.eval()
    bbox = model(img_transformed.to("cuda")).tolist()[0]
    bbox_dicts = {"x_1":bbox[0],"y_1":bbox[1],"width":bbox[2],"height":bbox[3]}

    drawer = BboxDrawer()
    drawer.process_img(image=image_resized,bbox=bbox_dicts,base=False)
