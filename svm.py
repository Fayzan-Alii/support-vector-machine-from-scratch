import numpy as np
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# MNIST Data Loader Class
class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return np.array(images), np.array(labels)

    def sample_data(self, X, y, percentage=0.05):
        """Samples a percentage of images per class (0-9)."""
        sampled_X, sampled_y = [], []
        for label in np.unique(y):
            indices = np.where(y == label)[0]
            sample_size = int(len(indices) * percentage)
            sampled_indices = np.random.choice(indices, sample_size, replace=False)
            sampled_X.append(X[sampled_indices])
            sampled_y.append(y[sampled_indices])
        return np.vstack(sampled_X), np.hstack(sampled_y)

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

# SVM Model from scratch
class SVM:
    def __init__(self, kernel='linear', C=1.0, degree=3, gamma=0.05, learning_rate=0.001, iterations=1000):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None  # Will be initialized later
        self.bias = 0

    def kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + 1) ** self.degree
        elif self.kernel == 'rbf':
            if len(X1.shape) == 1:
                X1 = X1.reshape(1, -1)
            distance = np.linalg.norm(X1[:, None] - X2, axis=2) ** 2
            return np.exp(-self.gamma * distance)

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get the number of features for initializing weights
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)  # Initialize weights with the correct size
        self.bias = 0

        # Training loop for gradient descent
        for _ in range(self.iterations):
            for i in range(n_samples):
                # Compute decision boundary and update weights if necessary
                condition = y_[i] * (np.dot(X[i], self.weights) - self.bias) < 1
                if condition:
                    self.weights += self.lr * (X[i] * y_[i] - 2 * (1 / self.iterations) * self.weights)
                    self.bias += self.lr * y_[i]
                else:
                    self.weights -= self.lr * 2 * (1 / self.iterations) * self.weights

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.bias)

# One-vs-Rest SVM for multi-class classification
class OneVsRestSVM:
    def __init__(self, kernel='linear', C=1.0, degree=3, gamma=0.05, learning_rate=0.001, iterations=1000):
        self.models = []
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.lr = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        for i in range(n_classes):
            y_binary = np.where(y == i, 1, -1)
            svm = SVM(kernel=self.kernel, C=self.C, degree=self.degree, gamma=self.gamma,
                      learning_rate=self.lr, iterations=self.iterations)
            svm.fit(X, y_binary)
            self.models.append(svm)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.argmax(predictions, axis=0)

# Visualizing misclassified digits
def visualize_misclassified(X_test, y_test, y_pred):
    misclassified_indices = np.where(y_test != y_pred)[0]
    for i in range(10):
        plt.imshow(X_test[misclassified_indices[i]].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[misclassified_indices[i]]}, Predicted: {y_pred[misclassified_indices[i]]}")
        plt.show()

# Performance metrics
def evaluate(y_test, y_pred, kernel):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nResults for Kernel: {kernel}")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Main function to run the SVM classifier
def main():
    # Set file paths based on your dataset location
    input_path = 'input'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # Load MNIST data
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                       test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Sample 50% of data from each class
    x_train, y_train = mnist_dataloader.sample_data(x_train, y_train, percentage=0.05)
    x_test, y_test = mnist_dataloader.sample_data(x_test, y_test, percentage=0.05)

    # Flatten the images and normalize them
    x_train = np.array([x.flatten() for x in x_train]) / 255.0
    x_test = np.array([x.flatten() for x in x_test]) / 255.0

    # # Define hyperparameter grids
    # param_grid = {
    #     'linear': {'C': [0.01, 0.1, 1, 10], 'learning_rate': [0.0001, 0.001, 0.01], 'iterations': [100, 500, 1000]},
    #     'poly': {'C': [0.01, 0.1, 1, 10], 'degree': [2, 3, 5, 10], 'learning_rate': [0.0001, 0.001, 0.01], 'iterations': [100, 500, 1000]},
    #     'rbf': {'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'learning_rate': [0.0001, 0.001, 0.01], 'iterations': [100, 500, 1000]}
    # }

    # # Iterate through each kernel and parameter combinations
    # for kernel, params in param_grid.items():
    #     print(f"\n--- Tuning for Kernel: {kernel} ---")
    #     if kernel == 'linear':
    #         for C in params['C']:
    #             for lr in params['learning_rate']:
    #                 for iter_count in params['iterations']:
    #                     print(f"\nTesting C={C}, learning_rate={lr}, iterations={iter_count}")
    #                     model = OneVsRestSVM(kernel=kernel, C=C, learning_rate=lr, iterations=iter_count)
    #                     model.fit(x_train, y_train)
    #                     y_pred = model.predict(x_test)
    #                     evaluate(y_test, y_pred, kernel)

    #     elif kernel == 'poly':
    #         for C in params['C']:
    #             for degree in params['degree']:
    #                 for lr in params['learning_rate']:
    #                     for iter_count in params['iterations']:
    #                         print(f"\nTesting C={C}, degree={degree}, learning_rate={lr}, iterations={iter_count}")
    #                         model = OneVsRestSVM(kernel=kernel, C=C, degree=degree, learning_rate=lr, iterations=iter_count)
    #                         model.fit(x_train, y_train)
    #                         y_pred = model.predict(x_test)
    #                         evaluate(y_test, y_pred, kernel)

    #     elif kernel == 'rbf':
    #         for C in params['C']:
    #             for gamma in params['gamma']:
    #                 for lr in params['learning_rate']:
    #                     for iter_count in params['iterations']:
    #                         print(f"\nTesting C={C}, gamma={gamma}, learning_rate={lr}, iterations={iter_count}")
    #                         model = OneVsRestSVM(kernel=kernel, C=C, gamma=gamma, learning_rate=lr, iterations=iter_count)
    #                         model.fit(x_train, y_train)
    #                         y_pred = model.predict(x_test)
    #                         evaluate(y_test, y_pred, kernel)

    # kernels = ['linear', 'poly', 'rbf']

    # # Iterate through each kernel
    # for kernel in kernels:
    #     print(f"Running SVM with {kernel} kernel...")
    #     if kernel == 'poly':
    #         svm_model = OneVsRestSVM(kernel=kernel, C=1.0, degree=10, learning_rate=0.0001, iterations=500)
    #     elif kernel == 'rbf':
    #         svm_model = OneVsRestSVM(kernel=kernel, C=1.0, gamma=0.1, learning_rate=0.0001, iterations=500)
    #     else:
    #         svm_model = OneVsRestSVM(kernel=kernel, C=1.0, learning_rate=0.0001, iterations=500)

    #     # Train the model
    #     svm_model.fit(x_train, y_train)

    #     # Predict on the test set
    #     y_pred = svm_model.predict(x_test)

    #     # Evaluate the model
    #     evaluate(y_test, y_pred, kernel)

    #     # Visualize misclassified digits
    #     visualize_misclassified(x_test, y_test, y_pred)

    kernel = 'linear'

    svm_model = OneVsRestSVM(kernel='kernel', C=1.0, learning_rate=0.0001, iterations=100)
        # Train the model
    svm_model.fit(x_train, y_train)

    # Predict on the test set
    y_pred = svm_model.predict(x_test)

    # Evaluate the model
    evaluate(y_test, y_pred, kernel)

    # Visualize misclassified digits
    visualize_misclassified(x_test, y_test, y_pred)

if __name__ == "__main__":
    main()
