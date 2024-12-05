import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x ** 2


def relu(x):
    return np.maximum(0, 0)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.weights = []
        self.biases = []
        self.a = []
        self.activations = {
            "sigmoid": (sigmoid, sigmoid_derivative),
            "tanh": (tanh, tanh_derivative),
            "relu": (relu, relu_derivative),
        }
        self.init_weights()

    def init_weights(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i + 1])))
            self.a.append(np.zeros(layers[i]))
        self.a.append(np.zeros(layers[-1]))

    def forward(self, x):
        self.a[0] = x
        activation_func = self.activations[self.activation][0]
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            self.a[i + 1] = activation_func(z)
        return self.a[-1]

    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        derivative_func = self.activations[self.activation][1]
        delta = self.a[-1] - y

        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.a[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * derivative_func(self.a[i])

    def train(self, x, y, epochs, learning_rate, text_widget):
        self.losses = []
        for epoch in range(epochs):
            output = self.forward(x)
            loss = np.mean((output - y) ** 2)
            self.losses.append(loss)
            self.backward(x, y, learning_rate)

            if epoch % 10 == 0 or epoch == epochs - 1:
                text_widget.insert(
                    tk.END, f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}\n"
                )
                text_widget.see(tk.END)
                text_widget.update()

    def test(self, x, y):
        predictions = self.forward(x)
        predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        return accuracy


def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        X = data.drop("Class", axis=1).values
        y = (data["Class"] == "Besni").astype(int).values.reshape(-1, 1)
        return X, y
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")
        return None, None


def start_training():
    try:
        input_size = int(input_nodes_entry.get())
        hidden_layers = list(map(int, hidden_nodes_entry.get().split(",")))
        output_size = int(output_nodes_entry.get())
        activation = activation_var.get()
        learning_rate = float(learning_rate_entry.get())
        epochs = int(epochs_entry.get())

        # Load dataset
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        X, y = load_dataset(file_path)
        if X is None or y is None:
            return

        # Normalize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the neural network
        global nn
        nn = NeuralNetwork(input_size, hidden_layers, output_size, activation)
        result_text.insert(tk.END, "Training started...\n")
        nn.train(X_train, y_train, epochs, learning_rate, result_text)

        # Test accuracy
        accuracy = nn.test(X_test, y_test)
        result_text.insert(tk.END, f"Training complete! Test Accuracy: {accuracy:.2f}\n")
        result_text.see(tk.END)

    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# GUI Setup
root = tk.Tk()
root.title("Neural Network Trainer")

tk.Label(root, text="Input Nodes:").grid(row=0, column=0, padx=5, pady=5)
input_nodes_entry = tk.Entry(root)
input_nodes_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Hidden Layers (comma-separated):").grid(row=1, column=0, padx=5, pady=5)
hidden_nodes_entry = tk.Entry(root)
hidden_nodes_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="Output Nodes:").grid(row=2, column=0, padx=5, pady=5)
output_nodes_entry = tk.Entry(root)
output_nodes_entry.grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="Activation Function:").grid(row=3, column=0, padx=5, pady=5)
activation_var = tk.StringVar(value="sigmoid")
tk.OptionMenu(root, activation_var, "sigmoid", "tanh", "relu").grid(row=3, column=1, padx=5, pady=5)

tk.Label(root, text="Learning Rate:").grid(row=4, column=0, padx=5, pady=5)
learning_rate_entry = tk.Entry(root)
learning_rate_entry.grid(row=4, column=1, padx=5, pady=5)

tk.Label(root, text="Epochs:").grid(row=5, column=0, padx=5, pady=5)
epochs_entry = tk.Entry(root)
epochs_entry.grid(row=5, column=1, padx=5, pady=5)

tk.Button(root, text="Start Training", command=start_training).grid(row=6, column=0, columnspan=2, pady=10)

result_text = scrolledtext.ScrolledText(root, width=60, height=20)
result_text.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
