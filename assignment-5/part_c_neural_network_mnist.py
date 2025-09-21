import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, sizes[i + 1])))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.relu(z)
            self.activations.append(activation)
            current_input = activation
        
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        output = self.softmax(z_output)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        gradients_w = []
        gradients_b = []
        
        error = output - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                dW = np.dot(self.activations[i].T, error) / m
                db = np.sum(error, axis=0, keepdims=True) / m
            else:
                error = np.dot(error, self.weights[i + 1].T) * self.relu_derivative(self.activations[i + 1])
                dW = np.dot(self.activations[i].T, error) / m
                db = np.sum(error, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
        
        return gradients_w, gradients_b
    
    def update_weights(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m
        return loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=True):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_train_loss = 0
            epoch_train_accuracy = 0
            
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                output = self.forward(X_batch)
                
                gradients_w, gradients_b = self.backward(X_batch, y_batch, output)
                self.update_weights(gradients_w, gradients_b)
                
                batch_loss = self.compute_loss(y_batch, output)
                epoch_train_loss += batch_loss
                
                predictions = np.argmax(output, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                epoch_train_accuracy += np.mean(predictions == true_labels)
            
            epoch_train_loss /= n_batches
            epoch_train_accuracy /= n_batches
            
            val_output = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_output)
            
            val_predictions = np.argmax(val_output, axis=1)
            val_true_labels = np.argmax(y_val, axis=1)
            val_accuracy = np.mean(val_predictions == val_true_labels)
            
            train_losses.append(epoch_train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(epoch_train_accuracy)
            val_accuracies.append(val_accuracy)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {epoch_train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}, Train Acc = {epoch_train_accuracy:.4f}, "
                      f"Val Acc = {val_accuracy:.4f}")
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy, predictions, true_labels

def load_mnist_data():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    X = mnist.data
    y = mnist.target.astype(int)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    
    return X, y

def preprocess_data(X, y, test_size=0.2, val_size=0.1):
    print("Preprocessing data...")
    
    X = X.astype(np.float32) / 255.0
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    y_onehot = np.eye(10)[y]
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_onehot, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=np.argmax(y_temp, axis=1)
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def visualize_predictions(X_test, y_true, y_pred, n_samples=10):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(n_samples):
        axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {y_true[i]}, Pred: {y_pred[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_architectures():
    print("Comparing Different Network Architectures")
    print("=" * 45)
    
    architectures = [
        ([128], "Single Hidden Layer (128)"),
        ([256], "Single Hidden Layer (256)"),
        ([128, 64], "Two Hidden Layers (128, 64)"),
        ([256, 128], "Two Hidden Layers (256, 128)"),
        ([512, 256, 128], "Three Hidden Layers (512, 256, 128)")
    ]
    
    X, y = load_mnist_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)
    
    results = []
    
    for hidden_sizes, name in architectures:
        print(f"\nTraining {name}...")
        
        nn = NeuralNetwork(
            input_size=784,
            hidden_sizes=hidden_sizes,
            output_size=10,
            learning_rate=0.01
        )
        
        train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
            X_train, y_train, X_val, y_val, epochs=50, verbose=False
        )
        
        test_accuracy, predictions, true_labels = nn.evaluate(X_test, y_test)
        
        results.append({
            'architecture': name,
            'test_accuracy': test_accuracy,
            'final_train_acc': train_accuracies[-1],
            'final_val_acc': val_accuracies[-1]
        })
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nArchitecture Comparison:")
    print("=" * 30)
    for result in results:
        print(f"{result['architecture']:30}: {result['test_accuracy']:.4f}")

def main():
    print("Feedforward Neural Network on MNIST Database")
    print("=" * 50)
    
    X, y = load_mnist_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)
    
    print("\nCreating Neural Network...")
    nn = NeuralNetwork(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        learning_rate=0.01
    )
    
    print("Network Architecture:")
    print("- Input Layer: 784 neurons (28x28 pixels)")
    print("- Hidden Layer 1: 256 neurons with ReLU activation")
    print("- Hidden Layer 2: 128 neurons with ReLU activation")
    print("- Output Layer: 10 neurons with Softmax activation")
    print("- Learning Rate: 0.01")
    print()
    
    print("Training the network...")
    train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
        X_train, y_train, X_val, y_val, epochs=100, batch_size=64
    )
    
    print("\nEvaluating on test set...")
    test_accuracy, predictions, true_labels = nn.evaluate(X_test, y_test)
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    plot_confusion_matrix(true_labels, predictions)
    
    print("Sample Predictions:")
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    visualize_predictions(X_test[sample_indices], true_labels[sample_indices], 
                         predictions[sample_indices])
    
    print("\n" + "=" * 50)
    compare_architectures()
    
    print("\nAnalysis:")
    print("- Neural network successfully learned to classify MNIST digits")
    print("- Training and validation curves show good learning progress")
    print("- Confusion matrix reveals which digits are most/least confused")
    print("- Different architectures show varying performance levels")

if __name__ == "__main__":
    main()
