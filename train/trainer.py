from base.base_trainer import BaseTrainer
from tqdm import tqdm
import torch


class VggTrainer(BaseTrainer):
    def __init__(self, model, loss, optimizer, metric, train_data_loader, valid_data_loader, gpu_mac=True, *args, **kwargs):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.gpu_mac = gpu_mac

        if self.gpu_mac:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = self.model.to(self.device)

    def _train_epoch(self, epoch):
        batch_loss = 0
        batch_total = 0
        batch_success = 0
        self.model.train()

        for inputs, labels in tqdm(self.train_data_loader):
            if self.gpu_mac:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_loss += loss.item() * inputs.size(0)
            batch_total, batch_success = self.metric(outputs, labels)

        epoch_loss = batch_loss / len(self.train_data_loader.dataset)
        epoch_accuracy = 100 * batch_success / batch_total

        return epoch_loss, epoch_accuracy

    def train(self, epochs):
        print(f"{epochs}epoch 학습 시작")

        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self._train_epoch(epoch)
            print(f"Epoch: {epoch} | Loss: {epoch_loss} | Accuracy: {epoch_accuracy}")

        print(f"{epochs}epoch 학습 종료")
        return epoch_loss, epoch_accuracy

    def valid(self):
        valid_loss = 0
        total = 0
        success = 0

        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.valid_data_loader:
                if self.gpu_mac:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = self.loss(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                valid_total, valid_success = self.metric(outputs, labels)

                total += valid_total
                success += valid_success

        valid_accuracy = 100 * success / total

        return valid_loss, valid_accuracy

