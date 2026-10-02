import numpy as np

def sigmoid(x): # x = 0 일때 최댓값 , 분모의 값이 작아질수록 값이 커지므로
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x): # 기본 relu (activation function)
    out = np.where( x > 0, x, 0)
    return out

def relu_derivative(x): # relu network -> 양수일 때 1, 0 또는 음수일때 0
    out =np.where( x > 0, 1, 0)
    return out

def update_grad(grad, func, layers): # layers 층 만큼 반복
    for i in range(layers):
        grad = grad * func

    return grad

layers = 10 # 은닉층 수
grad = 1.0 # 초기 기울기
# 기울기가 소실 되는 상황을 시뮬레이션 하기 위해 Sigmoid 도함수의 최댓값출력을 위해 0 , ReLU는 1(양수)값을 고정으로 입력하였음.

print(f"\n--- 기울기 소실 시뮬레이션 테스트 ---\n설정: 은닉층 수 = {layers}, 초기 기울기 = {grad:.1f}\n")
print(f"Sigmoid 네트워크 {layers}층 통과 후 최종 기울기 : {update_grad(grad, sigmoid_derivative(0), layers):.10f}")
print("결과 분석 : 기울기가 0에 가깝게 매우 작아져 기울기 소실(Gradient Vanishing)이 발생함을 확인할 수 있습니다.\n")

print(f"ReLU 네트워크 {layers}층 통과 후 최종 기울기 : {update_grad(grad, relu_derivative(1), layers):.1f}")
print(f"결과 분석 : 기울기가 {grad}으로 유지되어 깊은 층에서도 기울기 소실 없이 학습이 가능함을 확인할 수 있습니다.")