from main2 import MultilayerPerceptron
import numpy as np

P_IMPLICATION = MultilayerPerceptron(2, 3, 1)

P_IMPLICATION.gradient_descent(
    inputs = np.array(
       [[0, 0, 1, 1], [0, 1, 0, 1]]
    ).T,
    expected= np.array(
        [[1, 1, 0, 1]]
    ).T,
    learning_rate=1
)
print(
    P_IMPLICATION.run([0, 0]),
    P_IMPLICATION.run([0, 1]),
    P_IMPLICATION.run([1, 0]),
    P_IMPLICATION.run([1, 1])
)