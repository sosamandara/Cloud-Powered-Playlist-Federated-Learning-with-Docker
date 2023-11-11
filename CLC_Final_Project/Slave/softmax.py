import numpy as np
import math

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, num_classes=6):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_classes = num_classes
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros((num_features, self.num_classes))
        self.bias = np.zeros((1, self.num_classes))

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = softmax(linear_model)

            # Gradient descent updates
            dw = (1/num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/num_samples) * np.sum(y_predicted - y, axis=0)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = softmax(linear_model)
        return np.argmax(y_predicted, axis=1)


