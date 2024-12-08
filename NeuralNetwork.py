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
    return np.maximum(0, x)

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

    def train(self, x, y, epochs, learning_rate, text_widget, canvas):
        self.losses = []
        for epoch in range(epochs):
            output = self.forward(x)
            loss = np.mean((output - y) ** 2)
            self.losses.append(loss)
            self.backward(x, y, learning_rate)

            if epoch % 10 == 0 or epoch == epochs - 1:
                text_widget.insert(tk.END, f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}\n")
                text_widget.see(tk.END)
                text_widget.update()
                draw_network(canvas, self)

    def test(self, x, y):
        predictions = self.forward(x)
        predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        return accuracy

def draw_network(canvas, nn):
    canvas.delete("all")  
    layers = [nn.input_size] + nn.hidden_layers + [nn.output_size]
    max_nodes = max(layers)
    x_spacing = 400 // len(layers)
    y_spacing = 600 // max_nodes

    positions = []

    for layer_index, size in enumerate(layers):
        positions.append([])
        x = layer_index * x_spacing + 50
        for node_index in range(size):
            y = node_index * y_spacing + (y_spacing // 2)
            canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="blue")
            positions[-1].append((x, y))

    for layer_index, layer_positions in enumerate(positions[:-1]):
        for i, (x1, y1) in enumerate(layer_positions):
            for j, (x2, y2) in enumerate(positions[layer_index + 1]):
                weight = nn.weights[layer_index][i, j]
                color = "red" if weight < 0 else "green"
                width = min(5, abs(weight) * 10)
                canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

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

        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        X, y = load_dataset(file_path)
        if X is None or y is None:
            return

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        global nn
        nn = NeuralNetwork(input_size, hidden_layers, output_size, activation)
        draw_network(canvas, nn)

        result_text.insert(tk.END, "Training started...\n")
        nn.train(X_train, y_train, epochs, learning_rate, result_text, canvas)

        accuracy = nn.test(X_test, y_test)
        result_text.insert(tk.END, f"Training complete! Test Accuracy: {accuracy:.2f}\n")
        result_text.see(tk.END)

    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("Neural Network Trainer and Visualizer")

frame_left = tk.Frame(root)
frame_left.pack(side=tk.LEFT, padx=10, pady=10)

frame_right = tk.Frame(root)
frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

tk.Label(frame_left, text="Input Nodes:").grid(row=0, column=0, padx=5, pady=5)
input_nodes_entry = tk.Entry(frame_left)
input_nodes_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(frame_left, text="Hidden Layers (comma-separated):").grid(row=1, column=0, padx=5, pady=5)
hidden_nodes_entry = tk.Entry(frame_left)
hidden_nodes_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(frame_left, text="Output Nodes:").grid(row=2, column=0, padx=5, pady=5)
output_nodes_entry = tk.Entry(frame_left)
output_nodes_entry.grid(row=2, column=1, padx=5, pady=5)

tk.Label(frame_left, text="Activation Function:").grid(row=3, column=0, padx=5, pady=5)
activation_var = tk.StringVar(value="sigmoid")
tk.OptionMenu(frame_left, activation_var, "sigmoid", "tanh", "relu").grid(row=3, column=1, padx=5, pady=5)

tk.Label(frame_left, text="Learning Rate:").grid(row=4, column=0, padx=5, pady=5)
learning_rate_entry = tk.Entry(frame_left)
learning_rate_entry.grid(row=4, column=1, padx=5, pady=5)

tk.Label(frame_left, text="Epochs:").grid(row=5, column=0, padx=5, pady=5)
epochs_entry = tk.Entry(frame_left)
epochs_entry.grid(row=5, column=1, padx=5, pady=5)

tk.Button(frame_left, text="Start Training", command=start_training).grid(row=6, column=0, columnspan=2, pady=10)

result_text = scrolledtext.ScrolledText(frame_left, width=40, height=15)
result_text.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

canvas = tk.Canvas(frame_right, width=400, height=600, bg="white")
canvas.pack(padx=10, pady=10)

root.mainloop()
