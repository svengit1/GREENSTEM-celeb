from torch import nn


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
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

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
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2))

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

        self.pool5 = nn.MaxPool2d(kernel_size=(2,2))

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
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,4)

        self.flattening = [
            self.flatten,
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4
        ]

        self.block_array = {
            "block_one":self.block_one,
            "block_two":self.block_two,
            "block_three":self.block_three,
            "block_four":self.block_four,
            "block_five":self.block_five,
            "fc":self.flattening
        }

    def to_inline(self,device):
        for k in self.block_array.keys():
            for iter in range(len(self.block_array[k])):
                self.block_array[k][iter] = self.block_array[k][iter].to(device)



    def forward(self, x):
        for f in self.block_one:
            x= f(x)
        for f in self.block_two:
            x= f(x)
        for f in self.block_three:
            x = f(x)
        for f in self.block_four:
            x= f(x)
        for f in self.block_five:
            x = f(x)
        for f in self.flattening:
            x = f(x)
        return x