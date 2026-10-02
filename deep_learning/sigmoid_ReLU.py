import numpy as np

X = np.array([-2, -0.5, 0, 0.5, 2])

# 시그모이드 함수 및 도함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): # grad 계산에 필요
    return sigmoid(x) * (1 - sigmoid(x))

# Relu 함수 및 도함수

def relu(x):
    out = np.where( x > 0, x, 0)
    return out

def relu_derivative(x): # grad 계산에 필요
    out =np.where( x > 0, 1, 0)
    return out

#learning_rate = 0.005

print(f"\n입력값 x: \n{X}")
print('-' * 50)
print("1. Sigmoid 함수 테스트 결과")
print(f"sigmoid(x):\n{sigmoid(X)}")
print(f"sigmoid_derivative(x):\n{sigmoid_derivative(X)}")
print('-' * 50)
print("2. ReLU 함수 테스트 결과")
print(f"relu(x):\n{relu(X)}")
print(f"relu_derivative(x):\n{relu_derivative(X)}")