import sys
import torchvision
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_mnist_data(transform: transforms.Compose, batch_size: int = 32, val_split_ratio: float = 0.1) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the MNIST dataset and prepares train, validation and test DataLoaders.

    :param transform: Transformations to apply to the dataset.
    :param batch_size: Batch size for the training DataLoader. Validation and test DataLoaders use batch_size * 2. Defaults to 32.
    :param val_split_ratio: Ratio of the full training data to use for validation. Ensures at least one sample for validation. Defaults to 0.1.
    :raises ValueError: If the batch size is not positive or if the validation split ratio is not in the range (0, 1).
    :return: A tuple containing the train, validation and test DataLoaders.
    """
    if batch_size <= 0 or not (0 < val_split_ratio < 1):
        raise ValueError("Batch size must be positive and validation split ratio must be in the range (0, 1).")
    
    data_path = str(Path(sys.path[0]) / "src" / "data")
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    full_train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

    num_train_samples = len(full_train_dataset)
    val_size = max(1, int(val_split_ratio * num_train_samples))
    
    train_size = num_train_samples - val_size
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size * 2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader, test_loader