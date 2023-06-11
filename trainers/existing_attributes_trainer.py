import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from testing.cnn_class import MicroCNN
from tools.datasets import CelebADatasetParent, SubsetFactory
from tools.training_utilities import train, test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size_resized = (218, 218)
transform = transforms.Compose([
    transforms.Resize(size_resized),
    transforms.ToTensor(),
])

desired_celeba_attribute = 21


def transform_labels(labels):
    # Male label: -1 or 1
    labels = labels[desired_celeba_attribute]
    d = {-1: 0, 1: 1}
    labels = d[labels]
    return labels


class CelebADataset(CelebADatasetParent):

    def __getitem__(self, index):
        image, labels = super().__getitem__(index)
        labels = torch.nn.functional.one_hot(labels, num_classes=2)
        return image, labels


image_dir = "./img_celeba_bboxed2"
bbox_dir = "../Anno/list_attr_celeba.txt"

dataset = CelebADataset(image_dir, bbox_dir, image_transform=transform, label_transform=transform_labels)

batch_size = 32

train_size = 17000
val_size = 300
test_size = 300

subset_generator = SubsetFactory(train_size, val_size, test_size)

model = MicroCNN()
model.to_inline(device)
model.eval()

num_epochs = 20
save_interval = 3
save_dir = "GENDER_MODELS"

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def main():
    train_dataset, val_dataset, test_dataset = subset_generator(dataset, shuffle=False)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    validation_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                                        pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_data_loader, criterion, optimizer, device)
        validation_loss, validation_accuracy = test(model, validation_data_loader, criterion, device)
        print(
            f"\n Epoch {epoch + 1} out of {epoch - num_epochs} done \n Train loss: {train_loss} Train accuracy: {train_accuracy} "
            f"Validation loss: {validation_loss} Validation accuracy: {validation_accuracy} \n")
        if epoch + 1 % save_interval == 0:
            print("Saving!")
            torch.save(model, f"{save_dir}/save_epoch_{epoch}.pt")

    test_loss, test_accuracy = test(model, test_data_loader, criterion, device)
    print(f"Done! \n Test loss: {test_loss} Test Accuracy: {test_accuracy}")


if __name__ == '__main__':
    main()
