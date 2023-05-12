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
import time
from tqdm import tqdm
from testing.cnn_class import CNN

size_resized = (218,218)
device = torch.device("cuda")
## TRANSFROMS RESIZED IDE OBRNUTO ; TJ. Y,X!
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size_resized[::-1])])
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
            bbox = [size_resized[0]*bbox[0]/image_size[0], size_resized[1]*bbox[1]/image_size[1],
                   size_resized[0]*bbox[2]/image_size[0], size_resized[1]*bbox[3]/image_size[1]]
            bbox = torch.tensor(bbox, dtype=torch.float32)

        return image, bbox


image_dir = "./img_celeba"
bbox_dir = "./Anno/list_bbox_celeba.txt"
dataset = CelebADataset(image_dir, bbox_dir, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())

indices = np.arange(len(dataset))
# train_indices = indices[:9000]
# val_indices = indices[9000:9300]
# test_indices = indices[9300:]

train_size = 50000
val_size = 3000
test_size = 3000

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size+val_size]
test_indices = indices[train_size+val_size:train_size+val_size+test_size]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

trainloder = DataLoader(train_dataset, batch_size=bach_size, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=bach_size, shuffle=False, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=bach_size, shuffle=False, num_workers=8, pin_memory=True)



model = CNN()
model.to_inline(device)
loss_fn = nn.MSELoss()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epoch = 21
save_interval = 3

mse_saver = open('./mse_on_epoch.txt', 'w')

start_epoch = 0
def main():
    for epoch in range(start_epoch,n_epoch):
        print(f"{epoch} Broj epohe")
        t = time.perf_counter()
        mean_loss = 0
        for inputs, labels in tqdm(trainloder):
            inputs,labels = inputs.to(device, non_blocking=True),labels.to(device, non_blocking=True)
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            mean_loss += loss.item()/bach_size
            optimizer.step()

        l = mean_loss/len(trainloder)
        print(f"{l} mean loss")
        with open('./mse_on_epoch.txt', 'a') as f:
            f.write("epoch" + str(epoch) + "mean loss:" + str(l))
            f.write("\n")
            f.close()
        print(f"Time elapsed: {round(time.perf_counter() - t,2)} seconds")
        print(f"Percent done: {(epoch/n_epoch)*100}%")
        if epoch % save_interval == 0:
            print("SAVE")
            torch.save(model,f"BBOX_MODELS/modelEp{epoch}.pt")

    mse_saver.close()


if __name__ == '__main__':
    main()

# 178 × 218
