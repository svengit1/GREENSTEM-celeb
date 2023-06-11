from torch import nn
from torchvision import models


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.b1_act1 = nn.ReLU()
        self.b1_drop1 = nn.Dropout(0.1)

        self.b1_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.b1_act2 = nn.ReLU()
        self.b1_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.block_one = [
            self.b1_conv1,
            self.b1_act1,
            self.b1_drop1,
            self.b1_conv2,
            self.b1_act2,
            self.b1_pool2
        ]

        self.b2_conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.b2_act1 = nn.ReLU()
        self.b2_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.b2_act2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.block_two = [
            self.b2_conv1,
            self.b2_act1,
            self.b2_conv2,
            self.b2_act2,
            self.pool3
        ]

        self.b3_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.b3_act1 = nn.ReLU()
        self.b3_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.b3_act2 = nn.ReLU()
        self.b3_conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.b3_act3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.block_three = [
            self.b3_conv1,
            self.b3_act1,
            self.b3_conv2,
            self.b3_act2,
            self.b3_conv3,
            self.b3_act3,
            self.pool4
        ]

        self.b4_conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.b4_act1 = nn.ReLU()
        self.b4_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.b4_act2 = nn.ReLU()
        self.b4_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.b4_act3 = nn.ReLU()

        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2))

        self.block_four = [
            self.b4_conv1,
            self.b4_act1,
            self.b4_conv2,
            self.b4_act2,
            self.b4_conv3,
            self.b4_act3,
            self.pool5
        ]

        self.b4_conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.b4_act1 = nn.ReLU()
        self.b4_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.b4_act2 = nn.ReLU()
        self.b4_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.b4_act3 = nn.ReLU()

        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2))

        self.block_five = [
            self.b4_conv1,
            self.b4_act1,
            self.b4_conv2,
            self.b4_act2,
            self.b4_conv3,
            self.b4_act3,
            self.pool6
        ]

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)

        self.flattening = [
            self.flatten,
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4
        ]

        self.block_array = {
            "block_one": self.block_one,
            "block_two": self.block_two,
            "block_three": self.block_three,
            "block_four": self.block_four,
            "block_five": self.block_five,
            "fc": self.flattening
        }

    def to_inline(self, device):
        for k in self.block_array.keys():
            for iter in range(len(self.block_array[k])):
                self.block_array[k][iter] = self.block_array[k][iter].to(device)

    def forward(self, x):
        for f in self.block_one:
            x = f(x)
        for f in self.block_two:
            x = f(x)
        for f in self.block_three:
            x = f(x)
        for f in self.block_four:
            x = f(x)
        for f in self.block_five:
            x = f(x)
        for f in self.flattening:
            x = f(x)
        return x


class CNN_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1_conv1 = nn.Conv2d(3, 16, kernel_size=(8, 8), padding=6)
        self.b1_act1 = nn.ReLU()
        self.b1_drop1 = [nn.Dropout(0.3), nn.Dropout(0.3)]

        self.b1_conv2 = nn.Conv2d(16, 32, kernel_size=(8, 8), padding=6)
        self.b1_act2 = nn.ReLU()
        self.b1_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.block_one = [
            self.b1_conv1,
            self.b1_act1,
            self.b1_conv2,
            self.b1_act2,
            self.b1_pool2
        ]

        self.block_two = [
            self.b2_conv1,
            self.b2_act1,
            self.b2_conv2,
            self.b2_act2,
            self.b2_pool2,
            self.b2_conv3,
            self.b2_act3,
            self.b2_conv4,
            self.b2_act4,
            self.pool3
        ]

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(100352, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.flattening = [
            self.flatten,
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4
        ]

        self.block_array = {
            "block_one": self.block_one,
            "drop": self.b1_drop1,
            "block_two": self.block_two,
            "fc": self.flattening
        }

    def to_inline(self, device):
        for k in self.block_array.keys():
            for iter in range(len(self.block_array[k])):
                self.block_array[k][iter] = self.block_array[k][iter].to(device)

    def forward(self, x, train=True):
        if train:
            x = self.b1_drop1[0](x)
        for f in self.block_one:
            x = f(x)
        if train:
            x = self.b1_drop1[1](x)
        for f in self.block_two:
            x = f(x)
        for f in self.flattening:
            x = f(x)
        return x


class SoftWrappedCNN(CNN):
    def __init__(self):
        super().__init__()
        self.fc4 = nn.Linear(32, 2)
        self.flattening[-1] = self.fc4
        self.S = nn.Sigmoid()

    def forward(self, x):
        x = super().forward(x)
        return self.S(x)


class SoftWrapperCNNSmall(CNN_small):
    def __init__(self):
        super().__init__()
        self.fc4 = nn.Linear(64, 2)
        self.flattening[-1] = self.fc4
        self.sigm = nn.Sigmoid()

    def forward(self, x, train=True):
        x = super().forward(x)
        return self.sigm(x)


class MicroCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = nn.ParameterList([
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(4, 4)),
            nn.Flatten(),
            nn.Linear(20736, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 2),
            nn.Sigmoid()
        ])

    def to_inline(self, device):
        for layer_i in range(len(self.steps)):
            self.steps[layer_i] = self.steps[layer_i].to(device)

    def forward(self, x):
        for f in self.steps:
            x = f(x)
        return x


class CNN_small_age_gender(nn.Module):
    def __init__(self, soft=False):
        super().__init__()
        self.soft = soft
        self.b1_conv1 = nn.Conv2d(3, 16, kernel_size=(8, 8), padding=6)
        self.b1_act1 = nn.ReLU()
        self.b1_drop1 = [nn.Dropout(0.3), nn.Dropout(0.3)]

        self.b1_conv2 = nn.Conv2d(16, 32, kernel_size=(8, 8), padding=6)
        self.b1_act2 = nn.ReLU()
        self.b1_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.block_one = [
            self.b1_conv1,
            self.b1_act1,
            self.b1_conv2,
            self.b1_act2,
            self.b1_pool2
        ]

        self.b2_conv1 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.b2_act1 = nn.ReLU()
        self.b2_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.b2_act2 = nn.ReLU()
        self.b2_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.b2_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.b2_act3 = nn.ReLU()
        self.b2_conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.b2_act4 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=(2, 2))

        self.block_two = [
            self.b2_conv1,
            self.b2_act1,
            self.b2_conv2,
            self.b2_act2,
            self.b2_pool2,
            self.b2_conv3,
            self.b2_act3,
            self.b2_conv4,
            self.b2_act4,
            self.pool3
        ]

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(100352, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)
        self.flattening = [
            self.flatten,
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4,
            self.softmax
        ]

        self.block_array = {
            "block_one": self.block_one,
            "drop": self.b1_drop1,
            "block_two": self.block_two,
            "fc": self.flattening,
        }

    def to_inline(self, device):
        for k in self.block_array.keys():
            for iter in range(len(self.block_array[k])):
                self.block_array[k][iter] = self.block_array[k][iter].to(device)

    def forward(self, x, train=True):
        if train:
            x = self.b1_drop1[0](x)
        for f in self.block_one:
            x = f(x)
        if train:
            x = self.b1_drop1[1](x)
        for f in self.block_two:
            x = f(x)
        for f in self.flattening:
            x = f(x)
        if self.soft:
            for f in self.softmax:
                x = f(x)
        return x
