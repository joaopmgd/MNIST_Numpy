import numpy as np
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

def run():
    X_training, y_training, X_test, y_test = load_data()
    print(X_training.shape)
    print(y_training.shape)
    print(X_test.shape)
    print(y_test.shape)

if __name__ == '__main__':
    run()