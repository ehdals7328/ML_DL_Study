import numpy as np

# 1. 입력 데이터 (4x4 배열)
I = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# 2. 2x2 크기의 영역에서 가장 큰 값을 뽑아내기
def max_pooling2d(I):
    result = np.zeros((2, 2))
    for i in range(2):
        for k in range(2):
            window = I[std * i:std * i + 2, std * k:std * k + 2]
            result[i,k] = np.max(window)

    return result

# 3. Stride = 2
std = 2

# 4. 2x2 특징 맵을 반환하는 함수
print(max_pooling2d(I))