import torch
from abc import abstractmethod


class BaseTrainer:
    """
    Base class for all trainers
    """
    @abstractmethod
    def __init__(self, model, *args, **kwargs):
        self.model = model

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        epoch만큼 학습이 돌아갈 때 학습 full logic
        :param epoch: 학습이 돌아갈 epoch 수
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, epochs):
        raise NotImplementedError
