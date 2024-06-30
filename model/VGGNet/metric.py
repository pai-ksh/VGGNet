import torch


def vgg_accuracy(output, target):
    total = 0
    success = 0

    _, pred = torch.max(output.data, 1)

    total += target.size(0)
    success += (pred == target).sum().item()

    return total, success
