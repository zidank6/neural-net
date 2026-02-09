import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

def generate_spiral_data(points, classes):
    """
    Generates a 2D spiral dataset.
    points: Number of points per class
    classes: Number of classes (usually 2 or 3)
    """
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for j in range(classes):
        ix = range(points*j, points*(j+1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(j*4, (j+1)*4, points) + np.random.randn(points)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y

def generate_circles_data(n_samples=1000, noise=0.05, factor=0.5):
    """
    Generates concentric circles dataset using sklearn.
    """
    return make_circles(n_samples=n_samples, noise=noise, factor=factor)

def plot_decision_boundary(X, y, model, filename="decision_boundary.png"):
    """
    Plots the decision boundary of the model and saves it to a file.
    """
    print(f"Generating decision boundary plot: {filename}...")
    
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Flatten grid to feed into network
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions
    Z = model.feedforward(mesh_data)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=40, cmap=plt.cm.RdBu_r)
    plt.title(f"Decision Boundary")
    plt.savefig(filename)
    plt.close()
    print("Plot saved.")
