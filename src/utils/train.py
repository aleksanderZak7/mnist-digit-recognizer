import torch
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, accuracy_score

from src import ConvNeuralNetwork, set_random_seed, load_mnist_data, history_plot, confusion_matrix_plot

def train_model(device: torch.device, transform: transforms.Compose) -> ConvNeuralNetwork:
    """
    Orchestrates the full training process for the ConvNeuralNetwork on the MNIST dataset.

    :param device: The torch.device ('cuda' or 'cpu') to train the model on.
    :param transform: torchvision.transforms.Compose object with transformations for the MNIST dataset.
    :return: The trained ConvNeuralNetwork model.
    """
    set_random_seed()

    print("Loading MNIST data...")
    train_loader, val_loader, test_loader = load_mnist_data(transform)
    print(f"Loading complete.\n" 
          f"Train size: {len(train_loader.dataset)}, Validation size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")
    
    print("\nBeginning training:")
    model = ConvNeuralNetwork().to(device)
    model.fit(train_loader, val_loader)
    
    print("Training complete.\n")
    
    model.eval()
    model.save()
    history_plot(model.history)
    test_samples = torch.cat([images for images, _ in test_loader], dim=0).to(device)
    test_labels = torch.cat([labels for _, labels in test_loader], dim=0).to(device).numpy()
    
    y_pred_tensor: torch.Tensor = model.predict(test_samples)
    y_pred: np.ndarray = y_pred_tensor.to(device).numpy()
    
    classes: list[int] = [str(i) for i in range(10)]
    print(f"Test Accuracy: {accuracy_score(test_labels, y_pred):.4f}\n")
    print("Classification Report:\n", classification_report(test_labels, y_pred, target_names=classes))
    
    confusion_matrix_plot(test_labels, y_pred, classes)
    return model

def get_trained_model() -> tuple[ConvNeuralNetwork, transforms.Compose]:
    """
    Provides a trained ConvNeuralNetwork model for the MNIST dataset.

    :return: A tuple containing:
    - ConvNeuralNetwork model: The trained or loaded model.
    - transform: torchvision.transforms.Compose object with transformations for the MNIST dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    print("Initializing model...\n")
    model = ConvNeuralNetwork().to(device)
    
    if not model.load():
        model = train_model(device, transform)
        
    print("Model ready.")
    return model, transform