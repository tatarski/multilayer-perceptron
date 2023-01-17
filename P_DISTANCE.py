from main3 import *
import numpy as np


K = 10000
x = np.random.uniform(0, 10, K)
y = np.random.uniform(0, 10, K)

x_ = np.random.uniform(0, 10, K)
y_ = np.random.uniform(0, 10, K)

distances = np.array([np.linalg.norm([x[i] - x_[i], y[i] - y_[i]]) for i in range(K)])

P_Distance = MultilayerPerceptron([4, 5, 5, 1],
activation_function=RELU, activation_function_dZ=RELU_dZ)
P_Distance.gradient_descent(
    inputs = np.array(
       [x, y, x_, y_]
    ).T,
    expected= np.array(
        [distances]
    ).T,
    learning_rate=0.0000000001,
    iter_c = 1000,
    termination_threshhold = 10,
    change_learning_rate=True,
    learning_rate_increase=0.00000000001,
    error_delta_threshold=10
)
print(
    P_Distance.run([1, 1, 3, 3]),
    P_Distance.run([3, 3, 1, 1]),
)
# print(P_Distance.W)
# print(P_Distance.B)