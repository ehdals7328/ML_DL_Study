import numpy as np

I = np.array([1, 2, 3, 4, 5])
K = np.array([1, 0, -1])

out_len = len(I) - len(K) + 1
k_len = len(K)

result = np.zeros(3)

def conv1d(I, K):
    for i in range(out_len):
        result[i] = np.sum(I[i:i + k_len] * K)

    return result

conv1d(I,K)
print(result)