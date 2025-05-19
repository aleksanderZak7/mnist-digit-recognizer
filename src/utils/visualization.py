import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def history_plot(training_history: dict[str, list[float]], program_block: bool = False) -> None:
    """
    Plot training history for loss and accuracy.
    
    :param training_history: Dictionary containing training history with keys 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'.
    :param program_block: If True, the plot will block the program until closed, otherwise program execution continues while the plot is displayed.
    """
    plt.figure(figsize=(12, 4))
    metrics: tuple[str] = ('loss', 'accuracy')

    for idx, key in enumerate(metrics, start=1):
        plt.subplot(1, 2, idx)
        plt.plot(training_history[f'train_{key}'], label='Train')
        plt.plot(training_history[f'val_{key}'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel(key.capitalize())
        plt.title(f'{key.capitalize()} History')
        plt.legend()

    plt.tight_layout()
    plt.show(block=program_block)

def confusion_matrix_plot(y_test: np.ndarray, y_pred: np.ndarray, class_mapping: list[str], program_block: bool = True) -> None:
    """
    Plot confusion matrix with class labels.
    
    :param y_test: True labels for the test set.
    :param y_pred: Predicted labels for the test set.
    :param class_mapping: List of class names corresponding to the labels.
    :param program_block: If True, the plot will block the program until closed, otherwise program execution continues while the plot is displayed.
    """
    with plt.style.context('seaborn-v0_8-whitegrid'):
        cm: np.ndarray = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_mapping)
        cm_display.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show(block=program_block)

def show_mnist_digits(data_loader: DataLoader, program_block: bool = True) -> None:
    """
    Displays one sample image for each digit from the MNIST DataLoader in two rows with 5 digits per row.
    
    :param data_loader: DataLoader containing the MNIST data.
    :param program_block: If True, the plot will block the program until closed, otherwise program execution continues while the plot is displayed.
    :raises RuntimeError: If there are not enough samples to display all digits from 0 to 9.
    """
    shown_digits: set[int] = set()
    images_by_digit: dict[int, torch.Tensor] = {}

    for images, labels in data_loader:
        for img, label in zip(images, labels):
            label_int: int = int(label.item()) 
            if label_int not in shown_digits:
                images_by_digit[label_int] = img
                shown_digits.add(label_int)
                if len(shown_digits) == 10:
                    break
        if len(shown_digits) == 10:
            break

    if len(images_by_digit) < 10:
        raise RuntimeError("Not enough samples to display all digits from 0 to 9.")
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Sample MNIST Digits 0-9", fontsize=16)
    for idx, digit in enumerate(sorted(images_by_digit.keys())):
        row, col = idx // 5, idx % 5
        ax: Axes = axes[row, col]
        ax.imshow(images_by_digit[digit].squeeze(), cmap="gray")
        ax.set_title(f"Digit: {digit}")
        ax.axis("off")

    plt.tight_layout()
    plt.show(block=program_block)