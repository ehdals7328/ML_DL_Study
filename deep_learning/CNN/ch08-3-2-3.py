# CNN 핵심 파이프라인 구현하기

import numpy as np

def apply_relu(x):
    return np.maximum(0,x)

def max_pooling_2x2(input):
    pooling = np.zeros((2, 2))
    for i in range (2):
        for k in range(2):
            window = input[2 * i:2 * i+2, 2 * k:2 * k+2]
            pooling[i, k] = np.max(window)

    return pooling
            
# 1. 테스트용 입력 특징 맵 생성 (4x4 크기, 음수 값 포함)
test_feature_map = np.array([
    [1.5, -0.5, 2.0, -1.0],
    [-2.0, 3.0, 0.5, 1.0],
    [0.0, -1.5, 4.0, 2.5],
    [1.0, 2.0, -3.0,-2.5]
])

# 2. 문제 2: ReLU 활성화 함수 적용
relu_result = apply_relu(test_feature_map)

# 3. 문제 3: Max Pooling 적용 (Relu 통과 결과를 입력으로 사용)
pooling_result = max_pooling_2x2(relu_result)

# 4. 결과 출력
print("=== 1. 원본 특징 맵 (4x4) ===")
print(test_feature_map)

print("\n=== 2.ReLU 활성화 함수 적용 결과 (4x4) ===")
print("설명: 모든 음수 값이 0으로 치환되었습니다.")
print(relu_result)

print("\n=== 3.Max Pooling (2x2) 적용 결과 (2x2) ===")
print("설명: 2x2 영역마다 가장 큰 값을 추출하여 크기가 4x4에서 2x2로 압축 되었습니다.")
print(pooling_result)