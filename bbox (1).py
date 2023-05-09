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


# device = torch.device("cuda")
device = torch.device("mps")
device1 = torch.device("cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([218, 178])])
from torch.utils.data import DataLoader

bach_size = 32


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
            bbox = [178*bbox[0]/image_size[0], 218*bbox[1]/image_size[1],
                    178*bbox[2]/image_size[0], 218*bbox[3]/image_size[1]]
            # bbox = [218*bbox[0]/image_size[0],
            #         178*bbox[1]/image_size[1],
            #         bbox[2]*218/image_size[0],
            #         bbox[3]*178/image_size[1]]
            bbox = torch.tensor(bbox, dtype=torch.float32)

        return image, bbox


image_dir = "./celebA_mini"
bbox_dir = "./bbox_mini.txt"
dataset = CelebADataset(image_dir, bbox_dir, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

indeces = np.arange(len(dataset))
train_indices = indeces[:1000]
val_indices = indeces[1000:1300]
test_indices = indeces[1300:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

trainloder = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CNN(nn.Module):
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


model = CNN()
model.to(device)
loss_fn = nn.MSELoss()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epoch = 5
save_interval = 1
import time
mse_saver = open('./mse_on_epoch.txt', 'w')
for epoch in range(n_epoch):
    print(f"{epoch} Broj epohe")
    t = time.perf_counter()
    s = len(trainloder)
    p = 0
    mean_loss =0
    for inputs, labels in trainloder:
        p += 1
        # print(f"{((p/s) * 100):.2f}Postotak u epohi")
        inputs,labels = inputs.to(device),labels.to(device)
        labels = labels.to(torch.float32)
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        mean_loss+=loss.item()/bach_size
        optimizer.step()

    l = mean_loss/len(trainloder)
    print(f"{l} mean loss")
    with open('./mse_on_epoch.txt', 'w') as f:
        f.write("epoch" + str(epoch) + "mean loos:" + str(l))
        f.write("\n")
    print(f"Time elapsed: {round(time.perf_counter() - t,2)} seconds")
    print(f"Percent done: {(epoch/n_epoch)*100}%")
    if epoch % save_interval == 0:
        print("SAVE")
        torch.save(model,f"BBOX_MODELS/modelEp{epoch}.pt")

mse_saver.close()


# 178 × 218