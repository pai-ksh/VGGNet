from dataprovider.data_setter import CIFAR10DataSetter
from dataprovider.data_loader import CIFAR10DataLoder
from torch.utils.data import Subset
from model.VGGNet.loss import vgg_loss
from model.VGGNet.metric import vgg_accuracy
from model.VGGNet.model import VGG16, VGG19
from trainer import VggTrainer

import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np


def train(model, epochs, batch_size, num_classes, lr):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor()])

    train_data_setter = CIFAR10DataSetter(root='./data', train=True, download=True, transform=transform)
    train_subset_index = list(range(batch_size * 40))
    sub_train_dataset = Subset(train_data_setter, train_subset_index)
    train_data_loader = CIFAR10DataLoder(sub_train_dataset, batch_size=batch_size, shuffle=True)

    valid_data_setter = CIFAR10DataSetter(root='./data', train=False, download=True, transform=transform)
    valid_subset_index = list(range(batch_size * 10))
    sub_valid_dataset = Subset(valid_data_setter, valid_subset_index)
    valid_data_loader = CIFAR10DataLoder(sub_valid_dataset, batch_size=batch_size, shuffle=True)

    if model == "vgg16":
        model = VGG16(in_channels=3, num_classes=num_classes)
    elif model == "vgg19":
        model = VGG19(in_channels=3, num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    loss_fn = vgg_loss
    metric_fn = vgg_accuracy

    vgg_trainer = VggTrainer(model=model, loss=loss_fn, optimizer=optimizer, metric=metric_fn,
                             train_data_loader=train_data_loader, valid_data_loader=valid_data_loader, gpu_mac=True)

    train_loss, train_acc = vgg_trainer.train(epochs=epochs)
    print(f"Train Loss: {train_loss} | Train Accuracy: {train_acc}")
    valid_loss, valid_acc = vgg_trainer.valid()
    print(f"Valid Loss: {valid_loss} | Valid Accuracy: {valid_acc}")


if __name__ == "__main__":
    train(model="vgg16", epochs=10, batch_size=64, num_classes=10, lr=0.01)
