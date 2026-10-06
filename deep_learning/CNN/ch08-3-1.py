# CNN 핵심 파이프라인 구현하기

import numpy as np

# 1.테스트용 입력 이미지 생성 (5x5)
test_image = np.array([
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

# 2. 3x3 커널 정의 (라플라시안 필터)
test_kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

def zero_padd(image):
    arr = np.array()
    arr.append(np.pad(image, (1, 1)))
    return arr

def conv2d(I,K):
    I = np.pad(I, (1, 1))
    feature_map = np.zeros((5, 5))
    for i in range(5):
        for k in range(5):
            window = I[i:i+3, k:k+3]
            feature_map[i, k] = np.sum(window * K)

    return feature_map

# 3. 합성곱 연산 수행
result_feature_map = conv2d(test_image, test_kernel)

# 4. 결과 출력
print("===원본 입력 이미지 (5x5) ===")
print(test_image)

print("\n===적용된 커널 (3x3) ===")
print(test_kernel)

print("\n===합성곱 연산 결과 특징 맵 (5x5) ===")
print(result_feature_map)