#RNN 순전파 구현
import numpy as np

x = np.random.rand(4, 2, 1)
u = np.random.rand(3, 2)
w = np.random.rand(3, 3)
v = np.random.rand(1, 3)
h = np.zeros((3, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in range(4):
    print(f"{i+1} 번째 time step")
    h = np.tanh((u @ x[i]) + (w @ h))
    print(h)

    print("====")
    o = sigmoid(v @ h)
    print(o)