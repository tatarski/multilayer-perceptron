import numpy as np

def RELU(x):
    return max(0, x)

class NeuralNetwork:
    # neuron_count - 1d array. store neuron count for each layer
    
    def __init__(self, neuron_counts, tests):
        self.neuron_counts = neuron_counts
        self.W = [
            np.asmatrix([
                [np.random.uniform(0, 1) for j in range(0, neuron_counts[layer+1])]
                for i in range(0, neuron_counts[layer])
            ])
            for layer in range(0, len(neuron_counts) - 1)
        ]
        self.B = [
            np.asarray([np.random.uniform(0, 1) for i in range(0, neuron_counts[layer])])
            for layer in range(0, len(neuron_counts))
        ]
        # self.B[0] is not used

        # Store intermidiate values for last calculation
        self.A = []
        self.Z = []
        # self.Z[test_n][0] is not used
        # self.A[test_n][0] is placeholder for input vector

        self.tests = tests

    def initZ_A(self):
        return [
            np.asarray([0 for i in range(0, self.neuron_counts[layer])])
            for layer in range(0, len(self.neuron_counts))
        ]

    def clear_data(self):
        self.Z = []
        self.A = []

    # index used for storing processed data
    def run(self, input_vector):
        A = self.initZ_A()
        Z = self.initZ_A()
        A[0] = np.asarray(input_vector)
        for l in range(0, len(self.neuron_counts)-1):
            Z[l+1] = np.asarray(A[l]*self.W[l])[0][0] + self.B[l+1]
            A[l+1] = np.vectorize(RELU)(Z[l+1])
    
        self.Z.append(Z)
        self.A.append(A)
        return A[len(self.neuron_counts) - 1]

    def get_loss(self, target, output):
        sum = 0
        for i in range(len(target)):
            sum += (target[i] - output[i])**2
        return sum/len(target)
    
    def get_cost(self):
        sum = 0
        for i in range(len(self.tests)):
            sum += self.get_loss(
                self.tests[i][0],
                self.run(self.tests[i][1])
            )
        return sum/len(self.tests)

    # Gradient with respect to weights
    def dC_dW(self):
        sum = np.asarray([float(0), float(0)])
        for i in range(len(self.tests)):
            if self.A[i][1] > 0:
                sum += (self.A[i][1] - self.tests[i][0][0])*np.transpose(self.A[i][0])

        sum *= 2/len(self.tests)
        return sum

    # Gradient with respect to biases
    def dC_dB(self):
        sum = np.asarray([float(0)])
        for i in range(len(self.tests)):
            if self.A[i][1] > 0:
                sum += (self.A[i][1] - self.tests[i][0][0])

        sum *= 2/len(self.tests)
        return sum

    def gradient_descent(self):
        learning_rate = 0.01
        for i in range(10000):
            self.Z = []
            self.A = []
            cost = self.get_cost()
            print("CUR COST:", cost)
            print("W/B")
            print(self.W[0])
            print(self.B[1])
            dC_dW = np.transpose(np.asmatrix(self.dC_dW()))
            dC_dB = self.dC_dB()
            print("GRADIENT")
            print(dC_dW)
            print(dC_dB)
            self.W[0] -= learning_rate*dC_dW
            self.B[1] -= learning_rate*dC_dB

t =[
    [np.asarray([0]), [0,0]],
    [np.asarray([0]), [0,1]],
    [np.asarray([0]), [1,0]],
    [np.asarray([1]), [1,1]]
]
# t =[
#     [np.asarray([0]), [0,0]],
#     [np.asarray([1]), [0,1]],
#     [np.asarray([1]), [1,0]],
#     [np.asarray([1]), [1,1]]
# ]
a = NeuralNetwork([2,1], t)
a.gradient_descent()
print(a.run([1,1]))
print(a.run([1,0]))
print(a.run([0,0]))
print(a.run([0,1]))