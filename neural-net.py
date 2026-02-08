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
    A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
    """
    def __init__(self):
        # Weights and biases for hidden layer
        self.w_h1 = np.random.normal(size=2)  # weights for h1
        self.w_h2 = np.random.normal(size=2)  # weights for h2
        self.b_h1 = np.random.normal()        # bias for h1
        self.b_h2 = np.random.normal()        # bias for h2

        # Weights and biases for output layer
        self.w_o1 = np.random.normal(size=2)  # weights for o1
        self.b_o1 = np.random.normal()        # bias for o1

    def feedforward(self, x):
        # x is a list of 2 inputs
        # Use ReLU for hidden layer
        h1_input = np.dot(x, self.w_h1) + self.b_h1
        h1 = relu(h1_input)
        h2_input = np.dot(x, self.w_h2) + self.b_h2
        h2 = relu(h2_input)
        
        # Output from hidden layer becomes input to output layer
        # Output layer still uses Sigmoid for 0-1 probability
        o1 = neuron([h1, h2], self.w_o1, self.b_o1)
        return o1

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Feedforward ---
                h1_input = np.dot(x, self.w_h1) + self.b_h1
                h1 = relu(h1_input)
                h2_input = np.dot(x, self.w_h2) + self.b_h2
                h2 = relu(h2_input)
                
                o1_input = np.dot([h1, h2], self.w_o1) + self.b_o1
                o1 = sigmoid(o1_input)
                y_pred = o1

                # --- Calculate partial derivatives ---
                # d_L_d_w_pred: partial L / partial y_pred
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w_o1_h1 = h1 * sigmoid_derivative(o1_input)
                d_ypred_d_w_o1_h2 = h2 * sigmoid_derivative(o1_input)
                d_ypred_d_b_o1 = sigmoid_derivative(o1_input)

                d_ypred_d_h1 = self.w_o1[0] * sigmoid_derivative(o1_input)
                d_ypred_d_h2 = self.w_o1[1] * sigmoid_derivative(o1_input)

                # Neuron h1 (ReLU derivative)
                d_h1_d_w_h1_x1 = x[0] * relu_derivative(h1_input)
                d_h1_d_w_h1_x2 = x[1] * relu_derivative(h1_input)
                d_h1_d_b_h1 = relu_derivative(h1_input)

                # Neuron h2 (ReLU derivative)
                d_h2_d_w_h2_x1 = x[0] * relu_derivative(h2_input)
                d_h2_d_w_h2_x2 = x[1] * relu_derivative(h2_input)
                d_h2_d_b_h2 = relu_derivative(h2_input)

                # --- Update weights and biases ---
                # Neuron o1
                self.w_o1[0] -= learn_rate * d_L_d_ypred * d_ypred_d_w_o1_h1
                self.w_o1[1] -= learn_rate * d_L_d_ypred * d_ypred_d_w_o1_h2
                self.b_o1 -= learn_rate * d_L_d_ypred * d_ypred_d_b_o1

                # Neuron h1
                self.w_h1[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w_h1_x1
                self.w_h1[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w_h1_x2
                self.b_h1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b_h1

                # Neuron h2
                self.w_h2[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w_h2_x1
                self.w_h2[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w_h2_x2
                self.b_h2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b_h2

            if epoch % 100 == 0:
                y_preds = [self.feedforward(x) for x in data]
                # Correct MSE calculation for list
                loss = sum((yt - yp) ** 2 for yt, yp in zip(all_y_trues, y_preds)) / len(all_y_trues)
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
    
    # Train network
    network = NeuralNetwork()
    network.train(data, all_y_trues)

    # Make predictions
    emily = np.array([-7, -3]) # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches
    print(f"Emily: {network.feedforward(emily):.6f}")
    print(f"Frank: {network.feedforward(frank):.6f}")