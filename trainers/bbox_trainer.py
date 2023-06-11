import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from testing.cnn_class import CNN
from tools.datasets import CelebADatasetParent, SubsetFactory
from tools.training_utilities import train, test

size_resized = (218, 218)
device = torch.device("cuda")
# TRANSFROMS RESIZED IDE OBRNUTO ; TJ. Y,X!
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size_resized[::-1], antialias=True)])


def bbox_transform(labels, image):
    labels = labels[1:]
    image_size = image.size
    labels = [size_resized[0] * labels[0] / image_size[0], size_resized[1] * labels[1] / image_size[1],
              size_resized[0] * labels[2] / image_size[0], size_resized[1] * labels[3] / image_size[1]]
    return labels


class CelebADataset(CelebADatasetParent):
    pass


image_dir = "../img_celeba"
bbox_dir = "../Anno/existing/list_bbox_celeba.txt"
dataset = CelebADataset(image_dir, bbox_dir, image_transform=transform, label_transform=bbox_transform,
                        delim_whitespace=True)

batch_size = 32
train_size = 50000
val_size = 3000
test_size = 3000
subset_generator = SubsetFactory(train_size, val_size, test_size)

model = CNN()
model.to_inline(device)

num_epochs = 21
save_interval = 3
save_dir = "BBOX_MODELS"

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def main():

    train_dataset, val_dataset, test_dataset = subset_generator(dataset, shuffle=False)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    validation_data_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False, num_workers=8,pin_memory=True)
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
