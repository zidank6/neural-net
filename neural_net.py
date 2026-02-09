import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    # Subtract max for numerical stability (prevents e^big_number from exploding)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    # Clip predictions to avoid log(0) which would give -infinity
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Only sum over the "true" class (where y_true = 1)
    return -np.sum(y_true * np.log(y_pred_clipped)) / len(y_true)

def neuron(inputs, weights, bias):
    # weighted sum: w1*x1 + w2*x2 + ...
    total = np.dot(inputs, weights) + bias
    return sigmoid(total)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() if hasattr(y_true, 'mean') else ((y_true - y_pred) ** 2)

class NeuralNetwork:
    """
    A dynamic neural network.
    layer_sizes: List of integers, e.g. [784, 128, 64, 10].
    Uses Softmax output for multi-class (last layer > 1 neuron).
    Uses Sigmoid output for binary classification (last layer = 1 neuron).
    """
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Create weights and biases for each connection between layers
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i+1]
            
            # He initialization (good for ReLU)
            # Standard random normal scaled by sqrt(2/input_dim)
            weight_matrix = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/input_dim)
            bias_vector = np.zeros((1, output_dim))
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def feedforward(self, x):
        """Pass input through the network. Returns final activations."""
        current_activation = np.atleast_2d(x)
        
        # Hidden Layers (ReLU)
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            current_activation = relu(z)
            
        # Output Layer
        final_z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        
        # Use Softmax for multi-class, Sigmoid for binary
        if self.layer_sizes[-1] > 1:
            return softmax(final_z)
        else:
            return sigmoid(final_z)

    def train(self, data, all_y_trues, learn_rate=0.1, epochs=1000):
        '''
        data: (n x input_dim) numpy array
        all_y_trues: (n,) numpy array of integer labels OR one-hot encoded
        '''
        multi_class = self.layer_sizes[-1] > 1
        
        # Convert integer labels to one-hot encoding for multi-class
        if multi_class and all_y_trues.ndim == 1:
            num_classes = self.layer_sizes[-1]
            one_hot = np.zeros((len(all_y_trues), num_classes))
            one_hot[np.arange(len(all_y_trues)), all_y_trues.astype(int)] = 1
            all_y_trues = one_hot

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                x = np.atleast_2d(x)
                y_true = np.atleast_2d(y_true)
                
                # --- 1. Forward Pass (store activations for backprop) ---
                activations = [x]
                zs = []
                current_activation = x
                
                # Hidden Layers (ReLU)
                for i in range(len(self.weights) - 1):
                    z = np.dot(current_activation, self.weights[i]) + self.biases[i]
                    zs.append(z)
                    current_activation = relu(z)
                    activations.append(current_activation)
                
                # Output Layer
                final_z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
                zs.append(final_z)
                
                if multi_class:
                    y_pred = softmax(final_z)
                else:
                    y_pred = sigmoid(final_z)
                activations.append(y_pred)
                
                # --- 2. Backward Pass ---
                
                # Output layer delta
                # For Softmax + Cross-Entropy: delta = (y_pred - y_true)
                # For Sigmoid + MSE: delta = -2*(y_true - y_pred) * sigmoid'(z)
                # Both simplify nicely but for different mathematical reasons
                if multi_class:
                    delta = y_pred - y_true
                else:
                    error = -2 * (y_true - y_pred)
                    delta = error * sigmoid_derivative(final_z)
                
                # Propagate backwards through all layers
                for layer_idx in range(len(self.weights) - 1, -1, -1):
                    input_to_layer = activations[layer_idx]
                    
                    d_weights = np.dot(input_to_layer.T, delta)
                    d_biases = delta
                    
                    if layer_idx > 0:
                        prev_z = zs[layer_idx - 1]
                        delta = np.dot(delta, self.weights[layer_idx].T) * relu_derivative(prev_z)

                    self.weights[layer_idx] -= learn_rate * d_weights
                    self.biases[layer_idx] -= learn_rate * d_biases

            # Log progress at every 10% of training
            log_interval = max(1, epochs // 10)
            if epoch % log_interval == 0 or epoch == epochs - 1:
                preds = self.feedforward(data)
                if multi_class:
                    loss = cross_entropy_loss(all_y_trues, preds)
                    accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(all_y_trues, axis=1))
                    print(f"Epoch {epoch} | loss: {loss:.4f} | accuracy: {accuracy:.2%}")
                else:
                    loss = np.mean((all_y_trues - preds.flatten()) ** 2)
                    print(f"Epoch {epoch} | loss: {loss:.6f}")

    def save_weights(self, path):
        """Export all weights and biases to a JSON file."""
        import json
        model_data = {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        with open(path, 'w') as f:
            json.dump(model_data, f)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    # Quick test: binary classification (same as before)
    data = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
    labels = np.array([1, 0, 0, 1])
    
    network = NeuralNetwork([2, 4, 1])
    network.train(data, labels, epochs=1000)

    emily = np.array([-7, -3])
    frank = np.array([20, 2])
    print(f"Emily: {network.feedforward(emily).item():.6f}")
    print(f"Frank: {network.feedforward(frank).item():.6f}")