import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

def plot_activation_function(x, y, activation_name):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=activation_name)
    plt.title(activation_name + " Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(x, activation_function):
    if activation_function == "sigmoid":
        y = sigmoid(x)
        plot_activation_function(x, y, "Sigmoid")
    elif activation_function == "relu":
        y = relu(x)
        plot_activation_function(x, y, "ReLU")
    elif activation_function == "leaky_relu":
        y = leaky_relu(x)
        plot_activation_function(x, y, "Leaky ReLU")
    elif activation_function == "tanh":
        y = tanh(x)
        plot_activation_function(x, y, "Tanh")
    else:
        print("Invalid activation function. Please choose from sigmoid, relu, leaky_relu, or tanh.")

if __name__ == "__main__":
    x = np.linspace(-5, 5, 1000)  # Array of values from -5 to 5
    activation_function = input("Enter the activation function (sigmoid, relu, leaky_relu, tanh): ").lower()
    main(x, activation_function)