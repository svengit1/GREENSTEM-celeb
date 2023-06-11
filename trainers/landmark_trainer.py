import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from testing.cnn_class import CNN_small
from tools.datasets import CelebADatasetParent, SubsetFactory
from tools.training_utilities import train, test

size_resized = (218, 218)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize(size_resized),
    transforms.ToTensor(),
])


def label_transform(labels, image):
    labels = labels[1:]
    image_size = image.size
    labels = [size_resized[i % 2] * labels[i] / image_size[i % 2] for i in range(len(labels))]
    return labels


class CelebADataset(CelebADatasetParent):
    pass


image_dir = "../img_celeba"
bbox_dir = "../Anno/custom/list_landmarks_reshaped_celeba.txt"
dataset = CelebADataset(image_dir, bbox_dir, image_transform=transform, label_transform=label_transform,
                        df_index_key="Name")

batch_size = 32

train_size = 40000
val_size = 3000
test_size = 3000
subset_generator = SubsetFactory(train_size, val_size, test_size)

model = CNN_small()
model.to_inline(device)

num_epochs = 40
save_interval = 5
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
save_dir = "LANDMARK_MODELS"


def main():
    train_dataset, val_dataset, test_dataset = subset_generator(dataset, shuffle=False)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    validation_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                                        pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    for epoch in range(num_epochs):
        train_loss, _ = train(model, train_data_loader, criterion, optimizer, device)
        validation_loss, _ = test(model, validation_data_loader, criterion, device)
        print(f"\n Epoch {epoch + 1} out of {epoch - num_epochs} done \n Train loss: {train_loss} Train accuracy: Nan "
              f"Validation loss: {validation_loss} Validation accuracy: Nan \n")
        if epoch + 1 % save_interval == 0:
            print("Saving!")
            torch.save(model, f"{save_dir}/save_epoch_{epoch}.pt")

    test_loss,_ = test(model, test_data_loader, criterion, device)
    print(f"Done! \n Test loss: {test_loss} Test Accuracy: Nan")


if __name__ == '__main__':
    main()
