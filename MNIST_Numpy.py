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
    parameters_loaded = read_saved_weights()
    if parameters_loaded != None:
        W1 = parameters_loaded['W1'] 
        b1 = parameters_loaded['b1']
        W2 = parameters_loaded['W2']
        b2 = parameters_loaded['b2']
    else:
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

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[0]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = (1/m) * (np.dot(dZ2, A1.T))
    db2 = (1/m) * (np.sum(dZ2, axis = 1, keepdims = True))
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * (np.dot(dZ1, X.T))
    db1 = (1/m) * (np.sum(dZ1, axis = 1, keepdims = True))
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):

    W1 = parameters["W1"] - learning_rate * grads["dW1"]
    b1 = parameters["b1"] - learning_rate * grads["db1"]
    W2 = parameters["W2"] - learning_rate * grads["dW2"]
    b2 = parameters["b2"] - learning_rate * grads["db2"]
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def two_layer_model(X, Y, parameters, learning_rate = 0.001, num_iterations = 10000):
    grads = {}
    costs = []
    m = X.shape[0]

    for i in range(0, num_iterations):

        A, cache = forward_propagation(X, parameters)
        cost = compute_cost(A, Y)
        
        grads = backward_propagation(parameters, cache, X, Y)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    write_saved_weights(parameters)
    return parameters, costs

def predict(X, y, parameters):
    m = X.shape[1]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    predited_y = np.zeros((A2.shape))
    A2_predictions = np.argmax(A2, axis=0)

    for i in range(len(A2_predictions)):
        predited_y[A2_predictions[i],i] = y[A2_predictions[i],i]

    predictions = np.sum(predited_y)/m * 100
    return predictions

def plot_cost_history(cost_history):
    plt.plot(cost_history, color = 'blue')
    plt.title('Cost Reduction by Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

def read_saved_weights():
    try:
        return np.load('weights.npy').item()
    except FileNotFoundError:
        f = open('weights.npy', 'w')
        return None

def write_saved_weights(parameters):
    np.save('weights.npy', parameters)

def run():
    X_training, y_training, X_test, y_test = load_data()
    parameters = initialize_parameters(X_training.shape[0], 10, 10)
    #parameters, cost_history = two_layer_model(X_training, y_training, parameters)
    #plot_cost_history(cost_history)
    training_accuracy = predict(X_training, y_training, parameters)
    print("Training Accuracy = "+str(training_accuracy))
    test_accuracy = predict(X_test, y_test, parameters)
    print("Test Accuracy = "+str(test_accuracy))

if __name__ == '__main__':
    run()