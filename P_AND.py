from main2 import MultilayerPerceptron
import numpy as np
P_AND = MultilayerPerceptron(2, 3, 1)

P_AND.gradient_descent(
    inputs = np.array(
       [[0, 0, 1, 1], [0, 1, 0, 1]]
    ).T,
    expected= np.array(
        [[0, 0, 0, 1]]
    ).T,
    learning_rate=1
)
print(
    P_AND.run([0, 0]),
    P_AND.run([0, 1]),
    P_AND.run([1, 0]),
    P_AND.run([1, 1])
)

