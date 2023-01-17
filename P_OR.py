from main2 import MultilayerPerceptron
import numpy as np

P_OR = MultilayerPerceptron(2, 4, 1)

P_OR.gradient_descent(
    inputs = np.array(
       [[0, 0, 1, 1], [0, 1, 0, 1]]
    ).T,
    expected= np.array(
        [[0, 1, 1, 1]]
    ).T,
    learning_rate=1
)
print(
    P_OR.run([0, 0]),
    P_OR.run([0, 1]),
    P_OR.run([1, 0]),
    P_OR.run([1, 1])
)