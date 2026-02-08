import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def neuron(inputs, weights, bias):
    # weighted sum: w1*x1 + w2*x2 + ...
    total = sum(i * w for i, w in zip(inputs, weights)) + bias
    return sigmoid(total)

class NeuralNetwork:
    """
    A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
    """
    def __init__(self):
        # Weights and biases for hidden layer
        self.w_h1 = [random.uniform(-1, 1) for _ in range(2)]  # weights for h1
        self.w_h2 = [random.uniform(-1, 1) for _ in range(2)]  # weights for h2
        self.b_h1 = random.uniform(-1, 1)        # bias for h1
        self.b_h2 = random.uniform(-1, 1)        # bias for h2

        # Weights and biases for output layer
        self.w_o1 = [random.uniform(-1, 1) for _ in range(2)]  # weights for o1
        self.b_o1 = random.uniform(-1, 1)        # bias for o1

    def feedforward(self, x):
        # x is a list of 2 inputs
        h1 = neuron(x, self.w_h1, self.b_h1)
        h2 = neuron(x, self.w_h2, self.b_h2)

        # Output from hidden layer becomes input to output layer
        o1 = neuron([h1, h2], self.w_o1, self.b_o1)
        return o1

if __name__ == "__main__":
    test_values = [-10, -1, 0, 1, 10]
    
    print(f"{'Input':>10} | {'Sigmoid':>20}")
    print("-" * 35)
    for x in test_values:
        print(f"{x:>10} | {sigmoid(x):>20.10f}")

    print("\nDerivative test:")
    print(f"{'Input':>10} | {'Derivative':>20}")
    print("-" * 35)
    for x in test_values:
        print(f"{x:>10} | {sigmoid_derivative(x):>20.10f}")

    print("\nNeuron test:")
    inputs = [1.0, 2.0]
    weights = [0.5, -0.3]
    bias = 0.1
    output = neuron(inputs, weights, bias)
    print(f"Inputs: {inputs}, Weights: {weights}, Bias: {bias} -> Output: {output:.6f}")

    print("\nNeural Network (2 inputs, 2 hidden, 1 output) test:")
    network = NeuralNetwork()
    x = [2, 3]
    print(f"Input: {x} -> Output: {network.feedforward(x):.6f}")