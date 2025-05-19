from tqdm import tqdm
from pathlib import Path
from pickle import UnpicklingError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR


class ConvNeuralNetwork(nn.Module):
    """
    A Convolutional Neural Network for image classification.
    """
    def __init__(self) -> None:
        """Initializes the layers of the CNN and the model save path."""
        self._path = Path(__file__).parent / "model.pth"
        
        super().__init__()
        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self._conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self._bn2d1 = nn.BatchNorm2d(16)
        
        self._conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self._bn2d2 = nn.BatchNorm2d(32)
        
        self._dropout = nn.Dropout(0.20)
        
        self._dense1 = nn.Linear(32 * 7 * 7, 512)
        self._bn1 = nn.BatchNorm1d(512)
        
        self._dense2 = nn.Linear(512, 128)
        self._bn2 = nn.BatchNorm1d(128)
        
        self._output = nn.Linear(128, 10)
        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        :param x: Input tensor (e.g., batch of images).
        :return: Output tensor (logits for each class).
        """
        x = self._pool(torch.relu(self._bn2d1(self._conv1(x))))
        x = self._pool(torch.relu(self._bn2d2(self._conv2(x))))
        
        x = self._dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self._bn1(self._dense1(x)))
        x = torch.relu(self._bn2(self._dense2(x)))

        return self._output(x)
    
    def fit(self, train_data: DataLoader, val_data: DataLoader, epochs: int = 15, patience: int = 3) -> None:
        """
        Trains the model using the provided training and validation data.

        :param train_data: DataLoader for the training dataset.
        :param val_data: DataLoader for the validation dataset.
        :param epochs: Maximum number of epochs to train for. Defaults to 15.
        :param patience: Number of epochs to wait for improvement before early stopping. Defaults to 3.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ExponentialLR(optimizer, gamma=0.90)
        
        no_improve_epochs: int = 0
        best_val_loss: float = float('inf')
        device = next(self.parameters()).device
        best_model_state: dict = self.state_dict().copy()
        self._history: dict[str, list[float]] = { "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [] }

        for epoch in range(epochs):
            self.train()
            total_samples: int = 0
            current_loss: float = 0.0
            correct_predictions: int = 0

            with tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False) as tbar:
                for X_batch, y_batch in tbar:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    y_pred: torch.Tensor = self(X_batch)
                    loss: torch.Tensor = criterion(y_pred, y_batch)
                    
                    loss.backward()
                    optimizer.step()
                    
                    current_loss += loss.item()
                    _, predicted_labels = torch.max(y_pred, 1)
                    correct_predictions += (predicted_labels == y_batch).sum().item()
                    total_samples += y_batch.size(0)
                    tbar.set_postfix(loss=loss.item())
            
            avg_train_loss = current_loss / len(train_data)
            train_accuracy = correct_predictions / total_samples
            
            self._history["train_loss"].append(avg_train_loss)
            self._history["train_accuracy"].append(train_accuracy)
            
            self.eval()
            total_samples: int = 0
            current_loss: float = 0.0
            correct_predictions: int = 0

            with torch.no_grad():
                with tqdm(val_data, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False) as vbar:
                    for X_batch_val, y_batch_val in vbar:
                        X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
                        
                        y_pred_val_logits = self(X_batch_val)
                        val_loss_item: torch.Tensor = criterion(y_pred_val_logits, y_batch_val)
                        current_loss += val_loss_item.item()

                        _, predicted_val_labels = torch.max(y_pred_val_logits, 1)
                        correct_predictions += (predicted_val_labels == y_batch_val).sum().item()
                        total_samples += y_batch_val.size(0)
                        vbar.set_postfix(val_loss=val_loss_item.item())

            avg_val_loss = current_loss / len(val_data)
            val_accuracy = correct_predictions / total_samples
            
            self._history["val_loss"].append(avg_val_loss)
            self._history["val_accuracy"].append(val_accuracy)
            
            scheduler.step()
            
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            if avg_val_loss < best_val_loss:
                no_improve_epochs = 0
                best_val_loss = avg_val_loss
                best_model_state = self.state_dict().copy()
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs (patience: {patience}). Best validation loss: {best_val_loss:.4f}.\n")
                self.load_state_dict(best_model_state)
                return

        self.load_state_dict(best_model_state)

    def predict(self, x: torch.Tensor, with_confidence: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Makes predictions on the input tensor.

        :param x: Input tensor (e.g., a batch of images).
        :param with_confidence: If True, returns a tuple of (confidence scores, predicted labels). Otherwise, returns only predicted labels. Defaults to False.
        :return: Predicted class labels (torch.Tensor) or a tuple of (confidence scores, predicted labels) if `with_confidence` is True.
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            
            x = x.to(device)
            y_pred = self(x)
            return torch.max(torch.softmax(y_pred, 1), 1) if with_confidence else torch.argmax(y_pred, dim=1)
    
    def save(self) -> None:
        """Saves the model's state dictionary to the path defined in `self._path`."""
        torch.save(self.state_dict(), self._path)
    
    def load(self) -> bool:
        """
        Loads the model's state dictionary from the path defined in `self._path`.

        :return: True if loading was successful, False otherwise.
        """
        try:
            self.load_state_dict(torch.load(self._path, map_location=next(self.parameters()).device))
            self.eval()
            return True
        except (FileNotFoundError, RuntimeError, EOFError, UnpicklingError) as e:
            print(f"{type(e).__name__}: {e}\n")
            return False
            
    def _init_weights(self) -> None:
        """
        Initializes weights of the model layers.

        Uses Kaiming initialization for Conv2d and ReLU-activated Linear layers, Xavier for the final Linear layer and sets biases to zero (or one for BN weights).
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if layer.out_features == 10:
                    nn.init.xavier_normal_(layer.weight, gain=1.0)
                else:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
            elif isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.zeros_(layer.bias)
                nn.init.constant_(layer.weight, 1)
    
    @property
    def convolutional_layer(self) -> nn.Conv2d:
        """Returns the first convolutional layer of the model."""
        return self._conv1
    
    @property
    def history(self) -> dict[str, list[float]]:
        """
        Returns the training history (loss and accuracy for train/validation sets).

        :raises AttributeError: If the model has not been trained yet (history is not available).
        """
        if not hasattr(self, "_history"):
            raise AttributeError("History not available. Train the model first!")
        return self._history