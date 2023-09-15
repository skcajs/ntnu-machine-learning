import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random

class Adaline:
    def __init__(self, epochs=100, alpha = 0.1):
        self.epochs = epochs
        self.alpha = alpha
    
    def predict(self, inputs):
        net = np.dot(inputs, self.w[1:]) + self.w[0]
        return(np.where(net>=0, 1, -1))
    
    def train(self, train_inputs, train_labels):
        x = train_inputs
        t = train_labels
        self.cost = [] # to plot cost function over epochs
        self.w = random.rand(train_inputs.ndim+1)
        for _ in range(self.epochs):
            net = np.dot(x, self.w[1:]) + self.w[0]
            y = net # linear activation function
            error = (t - y) # veector
            # update weights using sum of gradients
            self.w[1:] += self.alpha * (np.dot(error, x)).mean()
            self.w[0] += self.alpha *  error.mean()
            cost = 0.5 * (error**2).sum()
            self.cost.append(cost)
        return self
    
if __name__ == "__main__":

    # Load IRIS data
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

    # extract first 100 class labels (50 iris-setosa and 50 iris-versicolor)
    x = df.iloc[0:100, [0,2]].values
    y = df.iloc[0:100,4].values
    y = np.where(y=='Iris-setosa', 1, -1)
    print(y)
    print(len(x))

    model = Adaline(epochs=50, alpha=0.0001)
    model.train(x,y)
    preds = model.predict(x)

    print(model.cost)
    print(model.w)
    print(preds)

    plt.plot(range(1, len(model.cost) + 1), model.cost, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training Error')
    plt.grid()
    plt.show()

    # scatter plot of preds
    plt.scatter(x[preds==1,0], x[preds==1,1], color='red', marker='o', label='setosa')
    plt.scatter(x[preds==-1,0], x[preds==-1,1], color='blue', marker='x', label='versicolor')

    # descision line
    xx1 = np.arange(x[:,0].min()-2, x[:,0].max()+2, 0.1)
    xx2 = -model.w[1]/model.w[2] * xx1 - model.w[0]/model.w[2]
    plt.plot(xx1, xx2, 'g--')
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Petal length (cm)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()