
import numpy as np
import matplotlib.pyplot as plt
import data_utils
import importlib.util
import sys
import os

# --- Import NeuralNet Module ---
# Dynamically import purely because the filename might have a dash/underscore issue
current_dir = os.path.dirname(os.path.abspath(__file__))
# Try loading neural_net.py first, if not found try neural-net.py
if os.path.exists(os.path.join(current_dir, "neural_net.py")):
    module_name = "neural_net"
    file_path = os.path.join(current_dir, "neural_net.py")
else:
    module_name = "neural-net"
    file_path = os.path.join(current_dir, "neural-net.py")

spec = importlib.util.spec_from_file_location(module_name, file_path)
nn_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = nn_module
spec.loader.exec_module(nn_module)
NeuralNetwork = nn_module.NeuralNetwork

def main():
    dataset_type = "spiral"  # Change to "circles" if needed
    
    print(f"Generating {dataset_type} dataset...")
    if dataset_type == "spiral":
        # Generate spiral data: 2 classes, 300 points per class
        X, y = data_utils.generate_spiral_data(points=300, classes=2) 
        # Spiral often needs slightly normalized coords to fit nicely in tanh/sigmoid range [-1, 1]
        # But our inputs are usually fine.
    else:
        X, y = data_utils.generate_circles_data(n_samples=600, noise=0.05, factor=0.5)

    # Initialize Neural Network
    # A spiral is hard! We need depth and width.
    # [2, 32, 32, 32, 1] is a good start for a complex spiral.
    print("Initialize Network: [2, 16, 16, 1]")
    network = NeuralNetwork([2, 16, 16, 1]) 
    
    # Train
    print("Training...")
    network.train(X, y)
    
    # Visualize
    data_utils.plot_decision_boundary(X, y, network, filename=f"decision_boundary_{dataset_type}.png")

if __name__ == "__main__":
    main()
