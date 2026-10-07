import numpy as np

feature_map = np.array([
    [1, 0, 5],
    [0, 2, 8],
    [3, 0, 7]
])

def flatten(x):
    return x.reshape(-1)

X = flatten(feature_map)

np.random.seed(42)
w = np.random.randn(9, 3)
b = np.random.randn(3)

def dense(X, w, b):
    result = (X @ w) + b

    return result

print(dense(X, w, b))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

print((softmax(dense(X, w, b))))
print(np.sum(softmax(dense(X, w, b))))