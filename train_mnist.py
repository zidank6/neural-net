import numpy as np
from sklearn.datasets import fetch_openml
from neural_net import NeuralNetwork

def load_mnist():
    """Download and prepare the MNIST dataset."""
    print("Downloading MNIST dataset (this may take a moment on first run)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X = mnist.data.astype(np.float64)
    y = mnist.target.astype(np.int64)
    
    # Normalize pixel values from 0-255 to 0-1
    X = X / 255.0
    
    # Split into train (60k) and test (10k) — standard MNIST split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    print(f"Image shape: {X_train.shape[1]} pixels (28x28 flattened)")
    
    return X_train, y_train, X_test, y_test

def evaluate(network, X_test, y_test):
    """Calculate accuracy on the test set."""
    predictions = network.feedforward(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == y_test)
    return accuracy

def main():
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Architecture: [784 inputs, 128 hidden, 64 hidden, 10 outputs]
    # This matches the reference visualization
    print("\nInitializing network: [784, 128, 64, 10]")
    network = NeuralNetwork([784, 128, 64, 10])
    
    # Train — MNIST is large, so we use a smaller learning rate
    # and fewer epochs than our toy examples
    print("Training...\n")
    network.train(X_train, y_train, learn_rate=0.01, epochs=10)
    
    # Evaluate on test set
    test_accuracy = evaluate(network, X_test, y_test)
    print(f"\nFinal test accuracy: {test_accuracy:.2%}")
    
    # Save trained weights for the web visualizer
    network.save_weights("model_weights.json")

if __name__ == "__main__":
    main()
