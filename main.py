import tkinter as tk
from src import get_trained_model, DigitRecognizerApp

if __name__ == "__main__":
    model, transform = get_trained_model()
    
    root = tk.Tk()
    print("Launching Tkinter application...")
    DigitRecognizerApp(root, model, transform)
    
    root.mainloop()