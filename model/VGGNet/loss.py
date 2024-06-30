import torch.nn as nn


def vgg_loss(output, target):
    cross_entropy_loss = nn.CrossEntropyLoss()
    return cross_entropy_loss(output, target)