from pathlib import Path
from torchmetrics import Metric
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, criterion, device):

        # input to constructor
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)

        # set using function set loader
        self.train_loader = None
        self.val_loader = None

        # set using function set metric
        self.train_metric = None
        self.val_metric = None

        # set using function set early stopping
        self.early_stopping_step = None

        # set using set checkpoint function
        self.save_best = None
        self.save_every_n_epochs = None
        self.save_last_epoch = None
        self.timestamp = None

        # set using set gradient clipping function
        self.clipping = None

        # updated during training
        self.best_score = None
        self.total_epochs = 0
        self.total_train_steps = 0
        self.total_val_steps = 0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.early_stopping_counter = 0
        self.early_stop = False
        # self.learning_rates = []

    def set_early_stopping(self, patience=5, delta=0):
        def early_stopping_step(val_loss):
            if self.best_score is not None:
                if val_loss >= self.best_score - delta:
                    self.early_stopping_counter += 1
                    print(
                        f'EarlyStopping counter: {self.early_stopping_counter} out of {patience}')
                    if self.early_stopping_counter >= patience:
                        self.early_stop = True
                else:
                    self.early_stopping_counter = 0
        self.early_stopping_step = early_stopping_step

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_metric(self, train_metric: Metric, val_metric: Metric = None):
        self.train_metric = train_metric
        self.val_metric = val_metric

    def set_checkpoint(self, save_path, save_best=True, save_every_n_epochs=None, save_last_epoch=False):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = save_path
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        self.save_last_epoch = save_last_epoch

    def set_gradient_clipping(self, clip_type, clip_value, norm_type=2):

        if clip_type.lower() == 'value':
            self.clipping = lambda: nn.utils.clip_grad_value_(
                self.model.parameters(), clip_value=clip_value
            )
        elif clip_type.lower() == 'norm':
            self.clipping = lambda: nn.utils.clip_grad_norm_(
                self.model.parameters(), clip_value, norm_type
            )
        else:
            raise ValueError(
                "Invalid clip_type provided. Use 'value' or 'norm'.")

    def _get_batch_metric(self, outputs, targets, multilabel, metric, threshold):

        if multilabel:
            outputs = (torch.sigmoid(outputs) > threshold).float()
        else:
            outputs = torch.argmax(outputs, dim=1)
        batch_metric = metric(outputs, targets)
        return batch_metric

    def _backward(self, loss, ):
        loss.backward()
        if callable(self.clipping):
            self.clipping()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _move_inputs_targets_to_gpu(self, inputs, targets):
        if isinstance(inputs, tuple):
            inputs = tuple(input_tensor.to(self.device)
                           for input_tensor in inputs)
        else:
            inputs = inputs.to(self.device)

        targets = targets.to(self.device)

        return inputs, targets

    def _log_print_epoch_loss_metric(self, train_loss, train_metric, val_loss, val_metric, epoch,
                                     num_epochs, dt_train, dt_valid):
        if self.train_metric:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                f"Train Metric: {train_metric:.4f}, Train Time: {dt_train}")

        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {val_loss:.4f}, Train Time: {dt_train}")

        if self.val_loader is not None:
            if self.val_metric:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, "
                    f"Val Metric: {val_metric:.4f}, Val Time: {dt_valid}")

            else:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Time: {dt_valid}")
        # print(f"Current Learning rate is {self.learning_rates[-1]}")
        print()

    def _run_batch(self, inputs, targets, metric, multilabel, threshold, training):

        inputs, targets = self._move_inputs_targets_to_gpu(
            inputs, targets)

        with torch.set_grad_enabled(training):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if training:
                self._backward(loss, )

        if metric is not None:
            batch_metric = self._get_batch_metric(
                outputs, targets, multilabel, metric, threshold)

        if training:
            self.total_train_steps += 1
        else:
            self.total_val_steps += 1

        return loss

    def _run_epoch(self, loader, training=True, multilabel=False, threshold=0.5):

        if training:
            self.model.train()
            metric = self.train_metric
        else:
            self.model.eval()
            metric = self.val_metric

        epoch_loss = 0.0
        num_samples = 0

        for i, (inputs, targets) in enumerate(loader):
            loss = self._run_batch(
                inputs, targets, metric, multilabel, threshold, training)

            epoch_loss += loss.item() * targets.size(0)
            num_samples += targets.size(0)

        epoch_loss /= num_samples

        if metric is not None:
            epoch_metric = metric.compute().item()
            metric.reset()
        else:
            epoch_metric = None

        return epoch_loss, epoch_metric

    def save_checkpoint(self, suffix=''):
        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"checkpoint_{self.timestamp}{suffix}.pt"
        checkpoint_data = {
            'total_epochs': self.total_epochs,
            'total_train_steps': self.total_train_steps,
            'total_val_steps': self.total_val_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_score,
        }

        # Add losses and metrics history to the checkpoint
        checkpoint_data['train_losses'] = self.train_losses
        checkpoint_data['val_losses'] = self.val_losses
        checkpoint_data['train_metrics'] = self.train_metrics
        checkpoint_data['val_metrics'] = self.val_metrics

        torch.save(checkpoint_data, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['total_epochs']
        self.total_train_steps = checkpoint['total_train_steps']

        if 'total_val_steps' in checkpoint:
            self.total_val_steps = checkpoint['total_val_steps']
        if 'val_loss' in checkpoint:
            self.best_score = checkpoint['val_loss']
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'train_metrics' in checkpoint:
            self.train_metrics = checkpoint['train_metrics']
        if 'val_metrics' in checkpoint:
            self.val_metrics = checkpoint['val_metrics']

        print(f"Loaded checkpoint from '{checkpoint_path}'.")

    def train(self, num_epochs, multilabel=False, threshold=0.5):
        assert self.train_loader is not None, "Train loader must be set before calling train()"

        if self.val_loader is None:
            print(
                'Validation loader is not set. The trainer will only execute training Loop')

        if all(value is None for value in [self.save_best, self.save_every_n_epochs, self.save_last_epoch]):
            print('Not saving any checkpoint')

        for epoch in range(num_epochs):

            t0 = datetime.now()

            train_loss, train_metric = self._run_epoch(
                self.train_loader, training=True, multilabel=multilabel, threshold=threshold)

            dt_train = datetime.now() - t0

            self.train_losses.append(train_loss)

            if self.train_metric:
                self.train_metrics.append(train_metric)

            if self.val_loader is not None:
                t0 = datetime.now()
                val_loss, val_metric = self._run_epoch(
                    self.val_loader, training=False, multilabel=multilabel,
                    threshold=threshold)
                dt_valid = datetime.now() - t0

                self.val_losses.append(val_loss)

                if self.val_metric:
                    self.val_metrics.append(val_metric)

                if callable(self.early_stopping_step):
                    self.early_stopping_step(val_loss)
                    if self.early_stop:
                        print("Early stopping triggered")
                        break

                if self.best_score is None or val_loss < self.best_score:
                    self.best_score = val_loss
                    self.best_epoch = self.total_epochs + 1
                    if self.save_best:
                        self.save_checkpoint(suffix=f'_best')

            # saving checkpoint

            if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint(
                    suffix=f'_epoch_{self.total_epochs + 1}')

            if self.save_last_epoch:
                self.save_checkpoint(
                    suffix=f'_last')

            self._log_print_epoch_loss_metric(train_loss, train_metric, val_loss, val_metric, epoch,
                                              num_epochs, dt_train, dt_valid)

            self.total_epochs += 1

    def plot_history(self):

        epochs = range(1, len(self.train_losses) + 1)

        plt.figure()
        plt.plot(epochs, self.train_losses, label="Train")
        plt.plot(epochs, self.val_losses, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        if self.train_metrics[0] is not None:
            plt.figure()
            plt.plot(epochs, self.train_metrics, label="Train")
            plt.plot(epochs, self.val_metrics, label="Validation")
            plt.xlabel("Epochs")
            plt.ylabel("Metric")
            plt.legend()
            plt.show()

    def predict(self, loader, return_targets=False, multilabel=False, threshold=0.5):

        self.model.to(self.device)
        self.model.eval()

        predictions = []
        targets_list = []

        with torch.no_grad():
            for inputs, targets in loader:

                inputs, targets = self._move_inputs_targets_to_gpu(
                    inputs, targets)

                outputs = self.model(inputs)

                if multilabel:
                    outputs = (torch.sigmoid(outputs) > threshold).float()
                else:
                    outputs = torch.argmax(outputs, dim=1)

                predictions.append(outputs.cpu())
                targets_list.append(targets.cpu())

        predictions_tensor = torch.cat(predictions, dim=0)
        targets_tensor = torch.cat(targets_list, dim=0)

        if return_targets:
            return predictions_tensor, targets_tensor,
        else:
            return predictions_tensor

    def sanity_check(self, num_classes):
        for inputs, targets in self.train_loader:

            inputs, targets = self._move_inputs_targets_to_gpu(
                inputs, targets)

            self.model.eval()
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            print(f'Actual loss: {loss}')
            break
        print(f'Expected Theoretical loss: {np.log(num_classes)}')
        self.model.train()

    @staticmethod
    def set_seed(seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
