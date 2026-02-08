import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def neuron(inputs, weights, bias):
    # weighted sum: w1*x1 + w2*x2 + ...
    total = sum(i * w for i, w in zip(inputs, weights)) + bias
    return sigmoid(total)

if __name__ == "__main__":
    test_values = [-10, -1, 0, 1, 10]
    
    print(f"{'Input':>10} | {'Sigmoid Output':>20}")
    print("-" * 35)
    for x in test_values:
        print(f"{x:>10} | {sigmoid(x):>20.10f}")

    print("\nDerivative test:")
    # Header for consistency
    print(f"{'Input':>10} | {'Derivative Output':>20}")
    print("-" * 35)
    for x in test_values:
        print(f"{x:>10} | {sigmoid_derivative(x):>20.10f}")

    print("\nNeuron test:")
    inputs = [1.0, 2.0]
    weights = [0.5, -0.3]
    bias = 0.1
    output = neuron(inputs, weights, bias)
    print(f"Inputs: {inputs}, Weights: {weights}, Bias: {bias}")
    print(f"Output: {output}")