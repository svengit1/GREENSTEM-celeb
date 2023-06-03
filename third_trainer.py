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
from testing.cnn_class import CNN_small

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
    def __init__(self, image_folder, bbox_file,label_transform=None, transform=None):
        self.image_folder = image_folder
        self.bbox_file = bbox_file
        self.transform = transform
        self.image_files = os.listdir(image_folder)[:10000]
        self.label_transform = label_transform
        self.bbox_df = pd.read_csv(bbox_file)

    def __len__(self):
        return len(self.bbox_df)

    def __getitem__(self, index):
        # Get bounding box for the current image
        row = self.bbox_df.loc[self.bbox_df['ID'] == index]
        #print(row)
        image_file = os.path.join(self.image_folder, row["Name"].values[0])
        image = Image.open(image_file).convert('RGB')
        labels = list(row.values[0])
        #image.show()
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            labels = self.label_transform(labels)
        #dictionary = {0:"white",1:"black"}
        #print(f"Label is: {dictionary[labels]}")

        return image, labels


image_dir = "./images_bboxed"
bbox_dir = "./quality_control/processed_SkinLabels_data.csv"

def label_transform(a):
    return a[2]

dataset = CelebADataset(image_dir, bbox_dir, transform=transform,label_transform=label_transform)
indices = np.arange(len(dataset))
# train_indices = indices[:9000]
# val_indices = indices[9000:9300]
# test_indices = indices[9300:]

batch_size = 32

train_size = 18000
val_size = 250
test_size = 250

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:train_size + val_size + test_size]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

trainloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# na čemu ce se model izvrsavati (cuda:0 - prva dostupna graficka kartica ili na CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN_small(soft=True)
model.to_inline(device)

# loss funkcija i optimizacija
criterion = nn.CrossEntropyLoss()
test_criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(net, trainloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    acc_total = 0.0
    for inputs,labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        labels = torch.reshape(labels,(batch_size,1))
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        acc = labels == outputs
        acc = torch.reshape(acc,(-1,)).tolist().count(True)/batch_size
        acc_total += acc
        running_loss += loss.item()

    return running_loss/len(trainloader),acc_total/len(trainloader)*100


def test(net, testloader, device):
    net.eval()
    running_loss = 0.0
    acc_total = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs,train=False)
            labels = torch.reshape(labels, (batch_size, 1))
            loss = test_criterion(outputs, labels.float())
            running_loss += loss.item()
            acc = labels == outputs
            acc = torch.reshape(acc, (-1,)).tolist().count(True) / batch_size
            acc_total += acc
    return running_loss / len(testloader),acc_total/len(trainloader)*100


# treniranje i testiranje kroz 20 epoha
def main():
    trainloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    num_epochs = 40
    for epoch in range(num_epochs):
        train_loss,train_acc = train(model, trainloader, criterion, optimizer, device)
        test_loss,test_acc = test(model, val_dataloader, device)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}% \n Test Loss: {test_loss:.4f}"
              f"Test Acc: {test_acc:.4f}%")
        if epoch %5 == 0 and epoch:
            print("Saving!")
            torch.save(model,f"RACE_MODELS/modelEp{epoch}.pt")

if __name__ == '__main__':
    main()
