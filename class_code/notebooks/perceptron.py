import numpy as np

class Perceptron:
    def __init__(self, dofs, epochs, alpha = 0.01):
        self.epochs = epochs
        self.alpha = alpha
        self.weights = np.zeros(dofs + 1)
        self.error = np.array([])
    
    def predict(self, inputs):
        net = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if net > 0:
            activation = 1
        else:
            activation = -1
        return activation
    
    def train(self, train_inputs, train_labels):
        for _ in range(self.epochs):
            error = 0
            for inputs, labels in zip(train_inputs, train_labels):
                preds = self.predict(inputs)
                self.weights[1:] += self.alpha * (labels - preds) * inputs
                self.weights[0] += self.alpha * (labels - preds)
                error += labels - preds
            self.error.append(error.mean())
