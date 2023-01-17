from main3 import MultilayerPerceptron
import numpy as np

P_XOR = MultilayerPerceptron([2, 3, 3, 1])

P_XOR.gradient_descent(
    inputs = np.array(
       [[0, 0, 1, 1], [0, 1, 0, 1]]
    ).T,
    expected= np.array(
        [[0, 1, 1, 0]]
    ).T,
    learning_rate=2,
    iter_c= 10000,
    termination_threshhold=0.000000000001,
)
print(
    P_XOR.run([0, 0]),
    P_XOR.run([0, 1]),
    P_XOR.run([1, 0]),
    P_XOR.run([1, 1])
)