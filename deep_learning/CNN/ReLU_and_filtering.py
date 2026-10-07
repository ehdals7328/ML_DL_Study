import numpy as np

feature_map = np.array([
    [ -2,   5,   0 ],
    [  3,  -8,  12 ],
    [ -1,  -4,   7 ]
])

# ReLU 연산을 적용하여 음수는 모두 0으로 바꾸고, 양수또는 0은 그대로 유지한 3x3 결과 반환

def relu(x):
    return np.maximum(0,x)

print(relu(feature_map))