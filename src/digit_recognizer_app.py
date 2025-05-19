import torch
import numpy as np
import tkinter as tk
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageTk
import torchvision.transforms as transforms

from src import ConvNeuralNetwork, GradCAM


class DigitRecognizerApp:
    """
    GUI application for handwritten digit recognition with Grad-CAM visualization.
    """
    __slots__ = ("_model", "_transform", "_device", "_grad_cam_generator", "_brush_size", "_canvas_width", "_canvas_height", "_line_color",
                 "_prediction_label", "_canvas", "_image", "_draw", "_saliency_display_canvas", "_saliency_display_photo")
    
    def __init__(self, root: tk.Tk, model: ConvNeuralNetwork, transform: transforms.Compose):
        """
        Initializes the Digit Recognizer application GUI and components.

        :param root: The main Tkinter window (tk.Tk instance).
        :param model: The pre-trained ConvNeuralNetwork model for digit recognition.
        :param transform: The torchvision.transforms.Compose object for preprocessing input images.
        """
        root.resizable(False, False)
        root.title("Digit Recognizer")
        
        self._model = model
        self._transform = transform
        self._device = next(self._model.parameters()).device
        self._grad_cam_generator = GradCAM(self._model, self._model.convolutional_layer)
        
        self._brush_size: int = 12
        self._canvas_width: int = 280
        self._canvas_height: int = 280
        self._line_color: str = "white"
        
        font: tuple[str, int] = ("Arial", 14)
        new_window_height = self._canvas_height + 200
        new_window_width = (self._canvas_width * 2) + 60 
        root.geometry(f"{new_window_width}x{new_window_height}") 

        self._prediction_label = tk.Label(root, text="Draw a digit and click 'Check'", font=("Arial", 16), width=new_window_width // 10, anchor="center")
        self._prediction_label.pack(pady=10)

        images_frame = tk.Frame(root)
        images_frame.pack(pady=10)

        drawing_frame = tk.Frame(images_frame)
        drawing_frame.pack(side=tk.LEFT, padx=10)

        drawing_label = tk.Label(drawing_frame, text="Your Drawing:", font=font)
        drawing_label.pack(pady=(0,5))

        self._canvas = tk.Canvas(drawing_frame, width=self._canvas_width, height=self._canvas_height, bg="black", cursor="crosshair")
        self._canvas.pack()
        self._canvas.bind("<B1-Motion>", self._paint)

        self._image = Image.new("L", (self._canvas_width, self._canvas_height), "black")
        self._draw = ImageDraw.Draw(self._image)

        saliency_frame = tk.Frame(images_frame)
        saliency_frame.pack(side=tk.LEFT, padx=10)
        
        saliency_display_label = tk.Label(saliency_frame, text="Grad-CAM Overlay:", font=font)
        saliency_display_label.pack(pady=(0,5))

        self._saliency_display_canvas = tk.Canvas(saliency_frame, width=self._canvas_width, height=self._canvas_height, bg="lightgrey")
        self._saliency_display_canvas.pack()
        self._saliency_display_photo: ImageTk.PhotoImage | None = None

        button_frame = tk.Frame(root)
        button_frame.pack(pady=20)

        check_button = tk.Button(button_frame, text="Check Number", command=self._predict_digit, font=font, width=12)
        check_button.pack(side=tk.LEFT, padx=10)

        clear_button = tk.Button(button_frame, text="Clear", command=self._clear_canvas, font=font, width=12)
        clear_button.pack(side=tk.LEFT, padx=10)
        
        self._perform_warmup()
    
    def _perform_warmup(self) -> None:
        """
        Performs a 'warm-up' pass for the model and Grad-CAM generator. 
        
        This can help reduce latency on the first actual prediction by initializing internal states or JIT compilations if any.
        """
        dummy_image: Image.Image = Image.new("L", (28, 28), 0)
        dummy_tensor: torch.Tensor = self._transform(dummy_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            _ = self._model(dummy_tensor)
            
        _ = self._grad_cam_generator(dummy_tensor, target_category_idx=0)

    def _paint(self, event):
        """
        Handles mouse drawing on the canvas.

        :param event: The Tkinter mouse event (contains x, y coordinates).
        """
        x1, y1 = (event.x - self._brush_size / 2), (event.y - self._brush_size / 2)
        x2, y2 = (event.x + self._brush_size / 2), (event.y + self._brush_size / 2)
        self._canvas.create_oval(x1, y1, x2, y2, fill=self._line_color, outline=self._line_color)
        self._draw.ellipse([x1, y1, x2, y2], fill=self._line_color, outline=self._line_color)

    def _clear_canvas(self):
        """Clears the drawing canvas, the underlying PIL image, the prediction text and the Grad-CAM display."""
        self._canvas.delete("all")
        self._draw.rectangle([0, 0, self._canvas_width, self._canvas_height], fill="black")
        self._prediction_label.config(text="Canvas cleared. Draw a digit.")
        
        self._saliency_display_canvas.delete("all")
        self._saliency_display_canvas.config(bg="lightgrey")

    def _predict_digit(self):
        """Processes the drawn image, gets a prediction from the model, updates the prediction label, and triggers Grad-CAM visualization."""
        original_drawn_pil_28x28: Image.Image = self._image.resize((28, 28), Image.Resampling.LANCZOS)
        
        img_tensor: torch.Tensor = self._transform(original_drawn_pil_28x28).unsqueeze(0)
        prediction_confidences, predicted_labels_tensor = self._model.predict(img_tensor, with_confidence=True) 
        
        confidence: float = prediction_confidences.item()
        predicted_label: int = predicted_labels_tensor.item()
        self._prediction_label.config(text=f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
        
        self._generate_and_display_gradcam_overlay(img_tensor, predicted_label, original_drawn_pil_28x28)

    def _generate_and_display_gradcam_overlay(self, input_tensor: torch.Tensor, predicted_class_idx: int, original_image_pil_28x28: Image.Image):
        """
        Generates the Grad-CAM heatmap, overlays it onto the original (28x28) image and displays the result on the saliency canvas.

        :param input_tensor: The preprocessed input tensor for the model.
        :param predicted_class_idx: The class index predicted by the model for this input.
        :param original_image_pil_28x28: The 28x28 PIL Image of the drawn digit (grayscale).
        """
        grad_cam_heatmap_np = self._grad_cam_generator(input_tensor, predicted_class_idx)
        heatmap_pil = Image.fromarray(grad_cam_heatmap_np).resize(original_image_pil_28x28.size, Image.Resampling.LANCZOS)

        alpha_blending: float = 0.8
        heatmap_np_resized: np.ndarray = np.array(heatmap_pil)
        colored_heatmap_rgba: np.ndarray = cm.jet(heatmap_np_resized) 
        colored_heatmap_rgba[:, :, 3] = alpha_blending
        
        original_display_pil: Image.Image = original_image_pil_28x28.convert("RGBA")
        heatmap_overlay_pil: Image.Image = Image.fromarray((colored_heatmap_rgba * 255).astype(np.uint8), "RGBA")
        blended_image_pil: Image.Image = Image.alpha_composite(original_display_pil, heatmap_overlay_pil)
        
        blended_image_for_display: Image.Image = blended_image_pil.convert("RGB").resize((self._canvas_width, self._canvas_height), Image.Resampling.NEAREST)

        self._saliency_display_photo = ImageTk.PhotoImage(blended_image_for_display)
        
        self._saliency_display_canvas.delete("all") 
        self._saliency_display_canvas.create_image(0, 0, anchor=tk.NW, image=self._saliency_display_photo)