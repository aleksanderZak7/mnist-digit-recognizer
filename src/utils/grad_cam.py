import torch
import numpy as np

from src.model import ConvNeuralNetwork


class GradCAM:
    """
    Implements Grad-CAM to produce a heatmap highlighting important regions in an image for a model's prediction for a specific class.
    """
    __slots__ = ("_model", "_device", "_target_layer", "_gradients", "_activations", "_fwd_hook_handle", "_bwd_hook_handle")
    
    def __init__(self, model: ConvNeuralNetwork, target_layer: torch.nn.Conv2d):
        """
        Initializes the GradCAM object.

        :param model: The ConvNeuralNetwork instance.
        :param target_layer: The specific convolutional layer to use for Grad-CAM.
        """
        self._model: ConvNeuralNetwork = model
        self._device = next(model.parameters()).device
        self._target_layer: torch.nn.Conv2d = target_layer
        
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._fwd_hook_handle: torch.utils.hooks.RemovableHandle = self._target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook_handle: torch.utils.hooks.RemovableHandle = self._target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: torch.nn.Module, input: tuple[torch.Tensor], output: torch.Tensor):
        """Hook to save the activations from the target layer's forward pass."""
        self._activations = output.to(self._device)

    def _save_gradients(self, module: torch.nn.Module, grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]):
        """Hook to save the gradients from the target layer's backward pass."""
        self._gradients = grad_output[0].to(self._device)

    def __call__(self, input_tensor: torch.Tensor, target_category_idx: int) -> np.ndarray:
        """
        Generates the Grad-CAM heatmap for the given input tensor and target category.

        :param input_tensor: Input image tensor, expected to be a single image.
        :param target_category_idx: Index of the target class for which to generate the heatmap.
        :return: A 2D NumPy array representing the normalized Grad-CAM heatmap.
        """
        self._model.zero_grad()
        output_logits: torch.Tensor = self._model(input_tensor.to(self._device))
        
        score: torch.Tensor = output_logits[0, target_category_idx]
        
        score.backward()
        gradients: torch.Tensor = self._gradients.detach()
        activations: torch.Tensor = self._activations.detach()
        pooled_gradients: torch.Tensor = torch.mean(gradients[0], dim=[1, 2])

        for i in range(activations.shape[1]):
            activations[0, i, :, :] *= pooled_gradients[i]
            
        heatmap: torch.Tensor = torch.sum(activations[0], dim=0)
        heatmap = torch.relu(heatmap)
        
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()