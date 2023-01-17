import numpy as np
import math

def sigmoid_function(x):
    return 1/(1 + math.e ** (-x))

# A - signe activation value
def sigmoid_dZ(Z, A):
    return A*(1-A)

def RELU(Z):
    return max(0, Z)

def RELU_dZ(Z, A):
    return 1 if Z >= 0 else 0

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
    def __init__(self, neuron_counts, activation_function = sigmoid_function, activation_function_dZ = sigmoid_dZ):
        self.activation_function = np.vectorize(activation_function)
        self.activation_function_dZ = np.vectorize(activation_function_dZ)
        self.neuron_counts = neuron_counts
        N = len(neuron_counts)
        self.W = [[] for i in range(N)]
        self.B = [[] for i in range(N)]
        for i in range(1, len(neuron_counts)):
            self.W[i] = np.random.uniform(low=0., high=1., size=(neuron_counts[i-1], neuron_counts[i]))
            self.B[i] = np.random.uniform(low=0., high=1., size=(1, neuron_counts[i]))

    def run(self, input):
        # Forward propagation
        N = len(self.neuron_counts)
        Z = [[] for _ in range(N)]
        A = [[] for _ in range(N)]
        A[0] = input
        for i in range(1, N):
            Z[i] = A[i-1] @ self.W[i] + self.B[i]
            A[i] = self.activation_function(Z[i])

        return A[len(self.neuron_counts) - 1]
        # return A[len(self.neuron_counts) - 1] > 0.5

    # inputs is np.array with dimensions (n_tests, n_inputs)
    # expected is np.array with dimensions (n_tests, n_outputs)
    # learning_rate is the speed of gradient descent
    def gradient_descent(self, inputs, expected, learning_rate, iter_c,
     termination_threshhold,
     change_learning_rate = False, error_delta_threshold = 1, learning_rate_increase = 0.001):
        N = len(self.neuron_counts)
        last_C = 0
        for j in range(iter_c):
            # Forward propagation
            Z = [[] for _ in range(N)]
            A = [[] for _ in range(N)]
            A[0] = inputs
            for i in range(1, N):
                Z[i] = A[i-1] @ self.W[i] + self.B[i]
                A[i] = self.activation_function(Z[i])

            C = cost(A[N-1], expected)
            print("MSE - ", C)
            print("MSE DELTA", C-last_C)
            # Increase learning delta if MSE delta is smaller than threshhold
            if (change_learning_rate and math.fabs(C - last_C) < error_delta_threshold):
                print("Increasing learning rate")
                learning_rate += learning_rate_increase
            
            # Terminate if error function is sufficiently minimized
            if (math.fabs(C) < termination_threshhold):
                print("THRESHOLD REACHED: END")
                break
            last_C = C
            print()
            
            # Backprop
            for i in range(N - 1, 0, -1):
                global dC_DZI_last
                if i == N - 1:
                    # Gradient of C with respect to pre-activation values of output layer
                    #           dC_dAI          dAI_dZI
                    dC_dZI = (A[i] - expected) * self.activation_function_dZ(Z[i], A[i])

                    # Keep dC_dZI for calculcations in prev layer
                    dC_DZI_last = dC_dZI

                    # Gradient of C with respect to weights to output layer
                    #            dZI_DWI      @ dC_DZI
                    dC_DWI = np.transpose(A[i-1]) @ dC_dZI
                    # print(dC_DWI)

                    # Gradient of C with respect to biases of output layer
                    #   dC_DBI = [1, 1, ..] @ dC_dZI

                    identity_vector = np.array([1 for i in range(len(dC_dZI))])
                    dC_DBI = identity_vector @ dC_dZI

                    # Adjust weights and biases of output layer
                    self.W[i] -= dC_DWI*learning_rate
                    self.B[i] -= dC_DBI*learning_rate
                else:
                    # Gradient of C with respect to pre-activation values of hidden layer
                    #          dC_dZ(I+1)   dZ(i+1)_dAi                     dAi_dZi
                    dC_dZI = (dC_DZI_last @ np.transpose(self.W[i+1])) * self.activation_function_dZ(Z[i], A[i])

                    # Keep dC_dZI for calculation in prev layer
                    dC_DZI_last = dC_dZI

                    # Gradient of C with respect to weights to hidden layer
                    #            dZI_DWI      @   dC_DZI
                    dC_DWI = np.transpose(A[i-1]) @ dC_dZI

                    # Gradient of C with respect to biases of hidden layer
                    #   dC_DBI = [1, 1, ..] @ dC_dZI
                    identity_vector = np.array([1 for i in range(len(dC_dZI))])
                    dC_DBI = identity_vector @ dC_dZI

                    # Adjust weights and biases of hidden layer
                    self.W[i] -= dC_DWI*learning_rate
                    self.B[i] -= dC_DBI*learning_rate