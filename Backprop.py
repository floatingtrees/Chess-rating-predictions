import numpy as np
import pickle
import time

#first is number of rows, then columns

def sigmoid(z): # Good
    s = 1/(1+np.exp(-z))
    return s

def relu(z): #Good
    s = np.clip(z, 0, None)
    assert np.max(-s) == 0, "clip not working"
    return s

def initialize(input_dim, neurons, L): # Good
    parameters = {}
    np.random.seed(1)
    parameters['W1'] = (2**(3/2)) * (np.random.rand(input_dim, neurons[0]) - 0.5)/np.sqrt(input_dim)
    parameters['b1'] = np.zeros((1, neurons[0]))
    for i in range(1, L):
        parameters["W" + str(i+1)] = (2**(3/2)) * (np.random.rand(neurons[i-1], neurons[i]) - 0.5)/np.sqrt(neurons[i-1])
        parameters['b' + str(i+1)] = np.zeros((1, neurons[i]))
    
    return parameters

def propagate(parameters, L, X, Y, test = False): 
    parameters['a0'] = X
    parameters['z1'] = np.matmul(X.T, parameters["W1"]) + parameters['b1'] 
    parameters['a1'] = relu(parameters['z1'])
    for i in range(2, L):
        parameters['z' + str(i)] = np.matmul(parameters['a' + str(i - 1)], parameters['W' + str(i)]) + parameters['b' + str(i)]
        parameters['a' + str(i)] = relu(parameters['z' + str(i)])
    parameters['z' + str(L)] = np.matmul(parameters['a' + str(L - 1)], parameters['W' + str(L)]) + parameters['b' + str(L)]
    parameters['a' + str(L)] = sigmoid(parameters['z' + str(L)])
    cost = -np.sum(np.multiply(Y.T, np.log(parameters['a' + str(L)])) + np.multiply((1-parameters['a' + str(L)]), 1-Y.T))/Y.shape[1]
    if test:
        print("\nTest accuracy: \n")
    print("Cost: ", cost)
    accuracy = (1 - (np.sum(np.abs(Y - np.around(parameters['a' + str(L)]).T)))/Y.shape[1]) * 100
    print("Accuracy", accuracy)
    return parameters, cost

def backpropagate(parameters, L, Y): #X.shape = (1, 750000) W.shape = (neurons from previous layer, neurons in current layer) 
    dcdz = Y.T - parameters['a' + str(L)]
    dzdw = parameters["a" + str(L - 1)].T
    parameters['dcdw' + str(L)] = np.matmul(dzdw, dcdz) / dzdw.shape[1]
    parameters['dcdb' + str(L)] = np.sum(dcdz)/Y.shape[1]
    dzda = parameters['W' + str(L)].T 
    dcda = np.matmul(dcdz, dzda) / dcdz.shape[1]
    assert dcda.shape == parameters["a" + str(L-1)].shape, dcda.shape
    for i in reversed(range(1, L)):
        dadz = np.heaviside(parameters['a' + str(i)], 0)
        dcdz = np.multiply(dadz, dcda)
        if i != 1:
            parameters['dcdw' + str(i)] = np.matmul(parameters['a' + str(i - 1)].T, dcdz) / dcdz.shape[0]
        else: 
            parameters['dcdw' + str(i)] = np.matmul(parameters['a' + str(i - 1)], dcdz) / dcdz.shape[0]
        parameters['dcdb' + str(i)] = np.sum(dcdz, axis = 0)/ dcdz.shape[0]
        if i != 1:
            dcda = np.matmul(dcdz, parameters["W" + str(i)].T) / dcdz.shape[0]


    return parameters

def update(parameters, learning_rate, L):
    for i in range(L):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] + parameters['dcdw' + str(i + 1)] * learning_rate
        parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] + parameters['dcdb' + str(i + 1)] * learning_rate
    return parameters

def optimize(parameters, L, X, Y, num_iterations, learning_rate):
    costs = []
    cd = []
    try:
        for i in range(num_iterations):
            parameters, cost = propagate(parameters, L, X, Y)
            costs.append(cost)
            parameters = backpropagate(parameters, L, Y)
            parameters = update(parameters, learning_rate, L)
        for i in range(num_iterations - 1):
            cd.append(costs[i - 1] - costs[i])
        print(cd)
    except KeyboardInterrupt:
        pass
    finally:
        return parameters

def model(neurons, X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    L = len(neurons)
    parameters = initialize(X_train.shape[0], neurons, L)
    parameters = optimize(parameters, L, X_train, Y_train, num_iterations, learning_rate)
    propagate(parameters, L, X_test, Y_test, test = True)

    return parameters

start = time.time()
# Data shapes line up
X_train = np.load('X_train_confirmed.npy') / 255
Y_train = np.load('Y_train_confirmed.npy')
X_test = np.load('X_test_confirmed.npy') / 255
print(X_train.shape, X_test.shape)
Y_test = np.load('Y_test_confirmed.npy')

neurons = [64, 16, 1]
print("\n PROGRAM STARTING \n")


parameters = model(neurons, X_train, Y_train, X_test, Y_test, num_iterations = 300, learning_rate = 0.05)
end = time.time()
print("Runtime: ", end - start)


f = open("deep_parameters_inv.pkl","wb")
pickle.dump(parameters,f)
f.close()


