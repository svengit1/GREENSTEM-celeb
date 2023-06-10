import math
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
from testing.cnn_class import MicroCNN

# standardizacija
size_resized = (218, 218)
transform = transforms.Compose([
    transforms.Resize(size_resized),
    transforms.ToTensor(),  # ne treba normalizirati, bolje je ostavljati u rgb formatu
])


# učitavanje podataka i Dataloader
# trainset = torchvision.datasets.CelebA(root='./podaci', split='train', transform=transform, download=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CelebA(root='./podaci', split='test', transform=transform, download=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Arhitektura


class CelebADataset(Dataset):
    def __init__(self, image_folder, bbox_file, transform=None):
        self.image_folder = image_folder
        self.bbox_file = bbox_file
        self.transform = transform
        self.image_files = os.listdir(image_folder)
        self.bbox_df = pd.read_csv(bbox_file, delim_whitespace=True,low_memory=False)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(image_file).convert('RGB')

        # Get bounding box for the current image
        row = self.bbox_df.loc[self.bbox_df['image_id'] == self.image_files[index]]
        dct = {-1:0,1:1}
        values = [dct[list(row.values[0])[20]], dct[list(row.values[0])[-1]]]
        # print("bbox for this image:", self.image_files[index], bbox)
        image_size = image.size
        if self.transform:
            # najoptimalnije sto sam se mogao sjetiti za sam resizing process
            labels =torch.tensor(values)
            image = self.transform(image)
        return image, labels


image_dir = "./img_celeba"
bbox_dir = "./Anno/list_attr_celeba.txt"
dataset = CelebADataset(image_dir, bbox_dir, transform=transform)
indices = np.arange(len(dataset))
# train_indices = indices[:9000]
# val_indices = indices[9000:9300]
# test_indices = indices[9300:]

batch_size = 32

train_size = 40000//2
val_size = 3000//2
test_size = 3000//2

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:train_size + val_size + test_size]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)



# na čemu ce se model izvrsavati (cuda:0 - prva dostupna graficka kartica ili na CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MicroCNN()
model.to_inline(device)

# loss funkcija i optimizacija
criterion = nn.BCELoss()
test_criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


def train(net, trainloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    acc_total = 0.0
    for inputs,labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        labels, outputs = labels.to(int), torch.round(outputs).to(int)
        acc = labels == outputs
        acc = torch.reshape(acc, (-1,)).tolist()
        acc = acc.count(True) / len(acc)
        acc_total += acc
        running_loss += loss.item()

        running_loss += loss.item()
    print(acc_total/len(trainloader))
    return running_loss/len(trainloader)


def test(net, testloader, criterion, device):
    net.eval()
    running_loss = 0.0
    acc_total = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = test_criterion(outputs, labels.float())
            running_loss += loss.item()
            labels, outputs = labels.to(int), torch.round(outputs).to(int)
            acc = labels == outputs
            acc = torch.reshape(acc, (-1,)).tolist()
            acc = acc.count(True) / len(acc)
            acc_total += acc
            running_loss += loss.item()
    print(acc_total/len(testloader))
    return running_loss / len(testloader)


# treniranje i testiranje kroz 20 epoha
def main():
    trainloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    num_epochs = 40
    for epoch in range(num_epochs):
        train_loss = train(model, trainloader, criterion, optimizer, device)
        test_loss = test(model, val_dataloader, criterion, device)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        if epoch %5 == 0 and epoch:
            print("Saving!")
            torch.save(model,f"AGE_GENDER_MODELS/modelEp{epoch}.pt")

if __name__ == '__main__':
    main()
