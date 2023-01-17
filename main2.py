import numpy as np
import math

def sigmoid_function(x):
    return 1/(1 + math.e ** (-x))

def activation_function(X):
    return np.vectorize(sigmoid_function)(X)

# Mean squared error
# O - list of output vectors
# Y - list of expected values 
def cost(outputs, expected):
    sum = 0
    N = len(expected)
    for i in range(N):
        sum += (outputs[i] - expected[i])**2
    return sum/N
    

class MultilayerPerceptron:
    def __init__(self, input_n, hidden_n, output_n):
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n

        self.W1 = np.random.uniform(low=0., high=1., size=(input_n,hidden_n))
        self.B1 = np.random.uniform(low=0., high=1., size=(1,hidden_n))
        self.W2 = np.random.uniform(low=0., high=1., size=(hidden_n,output_n))
        self.B2 = np.random.uniform(low=0., high=1., size=(1,output_n))

    def run(self, input):
        Z1 = input @ self.W1 + self.B1
        A1 = activation_function(Z1)
        Z2 = A1 @ self.W2 + self.B2
        A2 = activation_function(Z2)
        return A2 > 0.5

    # inputs is np.array with dimensions (n_tests, n_inputs)
    # expected is np.array with dimensions (n_tests, n_outputs)
    # learning_rate is the speed of gradient descent
    def gradient_descent(self, inputs, expected, learning_rate):

        for i in range(10000):
            Z1 = inputs @ self.W1 + self.B1
            A1 = activation_function(Z1)
            Z2 = A1 @ self.W2 + self.B2
            A2 = activation_function(Z2)

            C = cost(A2, expected)
            print("MSE - ", C)

            # Gradient of C with respect to pre-activation values of output layer
            #           dC_dA2         dA2_dZ2
            dC_dZ2 = (A2 - expected) * A2 * (1 - A2)

            # Gradient of C with respect to weights to output layer
            #            dZ2_DW2      @ dC_DZ2
            dC_DW2 = np.transpose(A1) @ dC_dZ2

            # Gradient of C with respect to biases of output layer
            #   dC_DB2 = [1, 1, ..] @ dC_dZ2

            identity_vector = np.array([1 for i in range(len(dC_dZ2))])
            dC_DB2 = identity_vector @ dC_dZ2


            # Adjust weights and biases of output layer
            self.W2 -= dC_DW2*learning_rate
            self.B2 -= dC_DB2*learning_rate

            # Gradient of C with respect to pre-activation values of hidden layer
            #          dC_dZ2   dZ2_dA1                 dA1_dZ1
            dC_dZ1 = (dC_dZ2 @ np.transpose(self.W2)) * A1 * (1-A1)

            # Gradient of C with respect to weights to hidden layer
            #           dZ1_dW1             dC_dZ1
            dC_dW1 = np.transpose(inputs) @ dC_dZ1

            # Gradient of C with respect to biases of hidden layer
            #   dC_DB1 = [1, 1, ..] @ dC_dZ1

            identity_vector = np.array([1 for i in range(len(dC_dZ1))])
            dC_DB1 = identity_vector @ dC_dZ1 

            # Adjust weights and biases of hidden layer
            self.W1 -= dC_dW1*learning_rate
            self.B1 -= dC_DB1*learning_rate


