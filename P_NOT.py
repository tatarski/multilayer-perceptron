from main2 import MultilayerPerceptron
import numpy as np

P_NOT = MultilayerPerceptron(1, 2, 1)

P_NOT.gradient_descent(
    inputs = np.array(
       [[0, 1]]
    ).T,
    expected = np.array(
        [[1, 0]]
    ).T,
    learning_rate=1
)
print(
    P_NOT.run([0]),
    P_NOT.run([1])
)