import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

def load_data():
    mndata = MNIST('./MNIST_data')
    training_set = np.asarray(mndata.load_training())
    test_set = np.asarray(mndata.load_testing())
    X_training = np.zeros((784, training_set.shape[1]))
    y_training = np.zeros((10, training_set.shape[1]))
    X_test = np.zeros((784, test_set.shape[1]))
    y_test = np.zeros((10, test_set.shape[1]))
    for i in range(len(training_set[1])):
        X_training[:,i] = np.asarray(training_set[0,i])/255
        y_training[np.asarray(training_set[1,i]),i] = 1
    for i in range(len(test_set[1])):
        X_test[:,i] = np.asarray(test_set[0,i])/255
        y_test[np.asarray(test_set[1,i]),i] = 1
    return X_training, y_training, X_test, y_test

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def sigmoid(Z):
    A =  1 / (1 + np.exp(-Z))    
    return A

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A, Y):
    m = Y.shape[1]
    cost = -(1/m) * np.sum(np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),1-Y))
    cost = np.squeeze(cost)
    return cost

def two_layer_model(X, Y, parameters, learning_rate = 0.001, num_iterations = 2):
    grads = {}
    costs = []
    m = X.shape[0]

    for i in range(0, num_iterations):

        A, cache = forward_propagation(X, parameters)
        cost = compute_cost(A, Y)
        
        if i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    return parameters, costs

def plot_cost_history(cost_history):
    plt.plot(cost_history, color = 'blue')
    plt.title('Cost Reduction by Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

def run():
    X_training, y_training, X_test, y_test = load_data()
    parameters = initialize_parameters(X_training.shape[0], 10, 10)
    parameters, cost_history = two_layer_model(X_training, y_training, parameters)
    plot_cost_history(cost_history)

if __name__ == '__main__':
    run()