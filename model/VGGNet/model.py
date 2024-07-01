import torch.nn as nn
from base.base_model import BaseModel


class ConvBlock2Layer(BaseModel):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class ConvBlock3Layer(BaseModel):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class ConvBlock4Layer(BaseModel):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class FcLayer(BaseModel):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=in_features, out_features=4096),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=4096, out_features=4096),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=4096, out_features=num_classes),
                                )

    def forward(self, x):
        x = self.fc(x)
        return x


class VGG16(BaseModel):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_block1 = ConvBlock2Layer(in_channels=in_channels, out_channels=64)
        self.conv_block2 = ConvBlock2Layer(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock3Layer(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock3Layer(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock3Layer(in_channels=512, out_channels=512)
        self.classifier = FcLayer(in_features=512*7*7, num_classes=num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.classifier(x.view(-1, 512*7*7))

        return x


class VGG19(BaseModel):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_block1 = ConvBlock2Layer(in_channels=in_channels, out_channels=64)
        self.conv_block2 = ConvBlock2Layer(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock4Layer(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock4Layer(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock4Layer(in_channels=512, out_channels=512)
        self.classifier = FcLayer(in_features=512*7*7, num_classes=num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.classifier(x.view(-1, 512*7*7))

        return x

# class VGG16(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # 첫번째 Conv 블럭
#         self.block1 = nn.Sequential( nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inaplace=True),
#                                      nn.MaxPool2d(kernel_size=2, stride=2)
#                                     )
#
#         # 두번째 Conv 블럭
#         self.block2 = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      nn.MaxPool2d(kernel_size=2, stride=2)
#                                    )
#
#         # 세번째 Conv 블럭
#         self.block3 = nn.Sequential( nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      #  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                                      #  nn.ReLU(inplace=True),
#                                      nn.MaxPool2d(kernel_size=2, stride=2)
#                                    )
#
#         # 네번째 Conv 블럭
#         self.block4 = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      #  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                                      #  nn.ReLU(inplace=True),
#                                      nn.MaxPool2d(kernel_size=2, stride=2)
#                                    )
#
#         # 다섯번째 Conv 블럭
#         self.block5 = nn.Sequential( nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                                      nn.ReLU(inplace=True),
#                                      #  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
#                                      #  nn.ReLU(inplace=True),
#                                      nn.MaxPool2d(kernel_size=2, stride=2)
#                                    )
#
#         # FC Layer
#         self.fc = nn.Sequential( nn.Linear(in_features=512*7*7, out_features=4096),
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(in_features=4096, out_features=4096),
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(in_features=4096, out_features=1000),
#                                  nn.Softmax(dim=1)
#                                 )
#
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.fc(x.view(-1, 512*7*7))
#         return x