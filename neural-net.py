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

def neuron(inputs, weights, bias):
    # weighted sum: w1*x1 + w2*x2 + ...
    total = np.dot(inputs, weights) + bias
    return sigmoid(total)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() if hasattr(y_true, 'mean') else ((y_true - y_pred) ** 2)

class NeuralNetwork:
    """
    A neural network that can be initialized with any number of layers.
    layer_sizes: A list of integers representing the number of neurons in each layer.
                 Example: [2, 3, 1] for 2 inputs, 3 hidden neurons, 1 output.
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
        """
        Passes input x through the network.
        x: Input vector (or batch of vectors)
        Returns: The final output of the network.
        """
        # Ensure x is 2D (1, input_dim) for consistent dot product
        current_activation = np.atleast_2d(x)
        
        # Iterate through all layers except the last one (Hidden Layers)
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            current_activation = relu(z)
            
        # Output Layer (Sigmoid activation for binary classification)
        final_z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        final_activation = sigmoid(final_z)
        
        return final_activation

    def train(self, data, all_y_trues):
        '''
        data: (n x input_dim) numpy array
        all_y_trues: (n,) numpy array of labels
        '''
        learn_rate = 0.1
        epochs = 1000 

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Ensure input is shaped correctly (1, input_dim)
                x = np.atleast_2d(x)
                
                # --- 1. Forward Pass (Store everything for Backprop) ---
                activations = [x] # List to store output of each layer. activations[0] is input.
                zs = []           # List to store "z" (input to activation function) for each layer
                
                current_activation = x
                
                # Hidden Layers (ReLU)
                for i in range(len(self.weights) - 1):
                    z = np.dot(current_activation, self.weights[i]) + self.biases[i]
                    zs.append(z)
                    current_activation = relu(z)
                    activations.append(current_activation)
                
                # Output Layer (Sigmoid)
                final_z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
                zs.append(final_z)
                y_pred = sigmoid(final_z)
                activations.append(y_pred)
                
                # --- 2. Backward Pass (Backpropagation) ---
                
                # Calculate error at output layer
                # d_Loss/d_z = (y_pred - y_true) for MSE with Sigmoid output if simplified?
                # Let's stick to the explicit chain rule you had:
                # d_L_d_ypred = -2 * (y_true - y_pred)
                # d_ypred_d_z = sigmoid_derivative(final_z)
                # delta = d_L_d_ypred * d_ypred_d_z
                
                error = -2 * (y_true - y_pred)
                delta = error * sigmoid_derivative(final_z) # Delta for the last layer
                
                # Store gradients to update after calculating all of them
                # (Or update in place as we go backwards)
                
                # We iterate backwards. `layer_idx` goes from last layer index down to 0
                for layer_idx in range(len(self.weights) - 1, -1, -1):
                    input_to_layer = activations[layer_idx]
                    
                    # Gradient of weights: dot product of input.T and delta
                    d_weights = np.dot(input_to_layer.T, delta)
                    d_biases = delta # Since input to bias is always 1
                    
                    # Calculate delta for the PREVIOUS layer (if we actally have a previous layer)
                    if layer_idx > 0:
                        # Propagate error backwards: delta * weights.T
                        # Then multiply by derivative of activation function of previous layer (ReLU)
                        prev_z = zs[layer_idx - 1]
                        delta = np.dot(delta, self.weights[layer_idx].T) * relu_derivative(prev_z)

                    # Update weights and biases for this layer
                    self.weights[layer_idx] -= learn_rate * d_weights
                    self.biases[layer_idx] -= learn_rate * d_biases

            if epoch % 100 == 0:
                # Calculate loss over entire dataset
                final_preds = self.feedforward(data) # This now returns shape (n, 1)
                # Flatten predictions for easy comparison with 1D y_trues
                loss = np.mean((all_y_trues - final_preds.flatten()) ** 2)
                print(f"Epoch {epoch} loss: {loss:.6f}")

if __name__ == "__main__":
    # ... (Keep previous tests if needed, or replace with training demo)
    
    # Define dataset
    data = np.array([
        [-2, -1],  # Alice
        [25, 6],   # Bob
        [17, 4],   # Charlie
        [-15, -6], # Diana
    ])
    all_y_trues = np.array([
        1, # Alice
        0, # Bob
        0, # Charlie
        1, # Diana
    ])
    
    # Train network with dynamic layers
    # [2 inputs, 4 hidden neurons, 1 output]
    network = NeuralNetwork([2, 4, 1])
    network.train(data, all_y_trues)

    # Make predictions
    emily = np.array([-7, -3]) # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches
    print(f"Emily: {network.feedforward(emily).item():.6f}")
    print(f"Frank: {network.feedforward(frank).item():.6f}")